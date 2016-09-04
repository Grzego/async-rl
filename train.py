from scipy.misc import imresize
from skimage.color import rgb2gray
from multiprocessing import *
from collections import deque
import queue
import gym
import numpy as np
import argparse

# -----
parser = argparse.ArgumentParser(description='Training model')
parser.add_argument('--game', default='Breakout-v0', help='OpenAI gym environment name', dest='game', type=str)
parser.add_argument('--processes', default=4, help='Number of processes that generate experience for agent',
                    dest='processes', type=int)
parser.add_argument('--lr', default=0.0001, help='Learning rate', dest='learning_rate', type=float)
parser.add_argument('--batch_size', default=64, help='Batch size to use during training', dest='batch_size', type=int)
parser.add_argument('--weight_switch', default=200, help='Number of batches before swapping network weights',
                    dest='weight_switch', type=int)
parser.add_argument('--checkpoint', default=0, help='Iteration to resume training', dest='checkpoint', type=int)
parser.add_argument('--save_rate', default=2000, help='Number of iterations before saving weights', dest='save_rate',
                    type=int)
parser.add_argument('--eps_decay', default=8000000,
                    help='Number of examples needed to decay epsilon to the lowest value', dest='eps_decay', type=int)
parser.add_argument('--lr_decay', default=2 * 80000000,
                    help='Number of examples needed to decay lr to the lowest value', dest='lr_decay', type=int)
parser.add_argument('--queue_size', default=256, help='Size of queue holding agent experience', dest='queue_size',
                    type=int)
parser.add_argument('--th_comp_fix', default=True,
                    help='Sets different Theano compiledir for each process', dest='th_fix', type=bool)
# -----
args = parser.parse_args()


# -----


def build_network(input_shape, output_shape):
    from keras.models import Sequential
    from keras.layers import InputLayer, Convolution2D, Flatten, Dense
    return Sequential([
        InputLayer(input_shape=input_shape),
        Convolution2D(16, 8, 8, subsample=(4, 4), activation='relu'),
        Convolution2D(32, 4, 4, subsample=(2, 2), activation='relu'),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(output_shape, activation='linear'),
    ])


# -----

class LearningAgent(object):
    def __init__(self, action_space, replay_size=32, screen=(84, 84), switch_rate=200):
        from keras.optimizers import RMSprop
        # -----
        self.screen = screen
        self.input_depth = 1
        self.past_range = 3
        self.observation_shape = (self.input_depth * self.past_range,) + self.screen
        self.replay_size = replay_size

        self.action_value_train = build_network(self.observation_shape, action_space.n)
        self.action_value = build_network(self.observation_shape, action_space.n)

        self.action_value_train.compile(optimizer=RMSprop(clipnorm=1.), loss='mse')
        self.action_value.compile(optimizer=RMSprop(clipnorm=1.), loss='mse')

        self.losses = deque(maxlen=100)
        self.q_values = deque(maxlen=100)
        self.switch_rate = switch_rate
        self.weight_switch = self.switch_rate
        self.unroll = np.arange(self.replay_size)
        self.counter = 0

    def learn(self, last_observations, actions, rewards, observations, not_terminals, discount=0.99,
              learning_rate=0.001):
        self.action_value_train.optimizer.lr.set_value(learning_rate)
        self.counter += 1
        # -----
        targets = self.action_value_train.predict_on_batch(last_observations)
        freeze_q_vals = self.action_value.predict_on_batch(observations)
        # -----
        # equation = rewards + not_terminals * discount * np.argmax(freeze_q_vals)
        rewards = np.clip(rewards, -1., 1.)
        equation = not_terminals
        equation *= np.max(freeze_q_vals, axis=1)
        equation *= discount
        targets[self.unroll, actions] = rewards + equation
        # -----
        loss = self.action_value_train.train_on_batch(last_observations, targets)
        self.losses.append(loss)
        self.q_values.append(np.mean(targets))
        print('\rIter: %8d; Lr: %8.7f; Loss: %7.4f; Min: %7.4f; Max: %7.4f; Avg: %7.4f --- Q-value; Min: %7.4f; Max: %7.4f; Avg: %7.4f' % (
                self.counter, learning_rate, loss, min(self.losses), max(self.losses), np.mean(self.losses),
                np.min(self.q_values), np.max(self.q_values), np.mean(self.q_values)), end='')
        self.weight_switch -= 1
        if self.weight_switch < 0:
            self.weight_switch = self.switch_rate
            self.action_value.set_weights(self.action_value_train.get_weights())
            return True
        return False


def learn_proc(mem_queue, weight_dict):
    import os
    pid = os.getpid()
    if args.th_fix:
        os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu,nvcc.fastmath=False,lib.cnmem=0,' + \
                                     'compiledir=th_comp_learn'
    # -----
    save_rate = args.save_rate
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    checkpoint = args.checkpoint
    lr_decay = args.lr_decay
    # -----
    env = gym.make(args.game)
    agent = LearningAgent(env.action_space, replay_size=args.batch_size, switch_rate=args.weight_switch)
    # -----
    if checkpoint > 0:
        agent.action_value_train.load_weights('model-%d.h5' % (checkpoint,))
        agent.action_value.set_weights(agent.action_value_train.get_weights())
    print(' %5d> Setting weights in dict' % (pid,))
    # -----
    weight_dict['update'] = 0
    weight_dict['weights'] = agent.action_value.get_weights()
    # -----
    last_obs = np.zeros((batch_size,) + agent.observation_shape)
    actions = np.zeros(batch_size, dtype=np.int32)
    rewards = np.zeros(batch_size)
    obs = np.zeros((batch_size,) + agent.observation_shape)
    not_term = np.zeros(batch_size)
    # -----
    index = 0
    counter = checkpoint
    agent.counter = counter / batch_size
    while True:
        last_obs[index, ...], actions[index], rewards[index], obs[index, ...], not_term[index] = mem_queue.get()
        # -----
        index = (index + 1) % batch_size
        if index == 0:
            lr = max(0.000000001, learning_rate * (1. - agent.counter * batch_size / lr_decay))
            updated = agent.learn(last_obs, actions, rewards, obs, not_term, learning_rate=lr)
            if updated:
                print(' %5d> Updating weights in dict' % (pid,))
                weight_dict['weights'] = agent.action_value.get_weights()
                weight_dict['update'] += 1
        # -----
        counter += 1
        if counter % (save_rate * batch_size) == 0:
            agent.action_value_train.save_weights('model-%d.h5' % (counter,), overwrite=True)


class ActingAgent(object):
    def __init__(self, action_space, screen=(84, 84)):
        from keras.optimizers import RMSprop
        # -----
        self.screen = screen
        self.input_depth = 1
        self.past_range = 3
        self.observation_shape = (self.input_depth * self.past_range,) + self.screen

        self.action_value = build_network(self.observation_shape, action_space.n)
        self.action_value.compile(optimizer=RMSprop(clipnorm=1.), loss='mse')

        self.action_space = action_space
        self.observations = np.zeros(self.observation_shape)
        self.last_observations = np.zeros_like(self.observations)

    def init_episode(self, observation):
        for _ in range(self.past_range):
            self.save_observation(observation)

    def sars_data(self, action, reward, observation, not_terminal):
        self.save_observation(observation)
        return self.last_observations, action, reward, self.observations, not_terminal

    def choose_action(self, epsilon=0.0):
        if np.random.random() < epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.action_value.predict(self.observations[None, ...]))

    def save_observation(self, observation):
        self.last_observations = self.observations[...]
        self.observations = np.roll(self.observations, -self.input_depth, axis=0)
        self.observations[-self.input_depth:, ...] = self.transform_screen(observation)

    def transform_screen(self, data):
        return rgb2gray(imresize(data, self.screen))[None, ...]


def generate_experience_proc(mem_queue, weight_dict, no, epsilon):
    import os
    pid = os.getpid()
    if args.th_fix:
        os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu,nvcc.fastmath=True,lib.cnmem=0,' + \
                                     'compiledir=th_comp_act_' + str(no)
    # -----
    moves = args.checkpoint
    procs = args.processes
    end = args.eps_decay
    batch_size = args.batch_size
    # -----
    print(' %5d> Process started with %6.3f' % (pid, epsilon))
    # -----
    env = gym.make(args.game)
    agent = ActingAgent(env.action_space)

    if moves > 0:
        print(' %5d> Loaded weights from file' % (pid,))
        agent.action_value.load_weights('model-%d.h5' % (moves,))
    else:
        import time
        while 'weights' not in weight_dict:
            time.sleep(0.1)
        agent.action_value.set_weights(weight_dict['weights'])
        print(' %5d> Loaded weights from dict' % (pid,))

    best_score = 0
    last_update = 0
    avg_score = deque(maxlen=20)

    while True:
        done = False
        episode_reward = 0
        noops = 0
        observation = env.reset()
        agent.init_episode(observation)

        # -----
        while not done:
            moves += procs
            decayed_epsilon = max(epsilon, epsilon + (1. - epsilon) * (end - moves) / end)
            action = agent.choose_action(decayed_epsilon)
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            best_score = max(best_score, episode_reward)
            # -----
            if action == 0:
                noops += 1
            else:
                noops = 0
            # -----
            if noops > 30:
                break
            # -----
            mem_queue.put(agent.sars_data(action, reward, observation, not done))
            # -----
            if moves % (2000 * procs) == 0:
                print(' %5d> Epsilon: %9.6f; Best score: %4d; Avg score: %6.2f' % (
                    pid, decayed_epsilon, best_score, np.mean(avg_score)))
            if moves % (batch_size * procs) == 0:
                update = weight_dict.get('update', 0)
                if update > last_update:
                    last_update = update
                    print(' %5d> Getting weights from dict' % (pid,))
                    agent.action_value.set_weights(weight_dict['weights'])
        # -----
        avg_score.append(episode_reward)


def init_worker():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main():
    manager = Manager()
    weight_dict = manager.dict()
    mem_queue = manager.Queue(args.queue_size)

    eps = [0.1, 0.05, 0.3, 0.5]
    pool = Pool(args.processes + 1, init_worker)

    try:
        for i in range(args.processes):
            pool.apply_async(generate_experience_proc, (mem_queue, weight_dict, i, eps[i % len(eps)]))

        pool.apply_async(learn_proc, (mem_queue, weight_dict))

        pool.close()
        pool.join()

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()


if __name__ == "__main__":
    main()
