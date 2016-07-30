from scipy.misc import imresize
from skimage.color import rgb2gray
from keras.models import Sequential
from keras.layers import InputLayer, Convolution2D, Flatten, Dense
from keras.optimizers import RMSprop
from multiprocessing import Process, Queue, Manager, Lock
from collections import deque
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
parser.add_argument('--queue_size', default=256, help='Size of queue holding agent experience', dest='queue_size',
                    type=int)
# -----
args = parser.parse_args()


# -----


def build_network(input_shape, output_shape):
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
    def __init__(self, action_space, replay_size=32, screen=(84, 84), compilation_lock=None, switch_rate=200):
        # from keras.models import Sequential
        # from keras.layers import InputLayer, Convolution2D, Flatten, Dense
        # from keras.optimizers import RMSprop

        self.screen = screen
        self.input_depth = 1
        self.past_range = 3
        self.observation_shape = (self.input_depth * self.past_range,) + self.screen
        self.replay_size = replay_size

        self.action_value_train = build_network(self.observation_shape, action_space.n)
        self.action_value = build_network(self.observation_shape, action_space.n)

        self.action_value_train.compile(optimizer=RMSprop(clipnorm=1.), loss='mse')
        self.action_value.compile(optimizer=RMSprop(clipnorm=1.), loss='mse')

        # Looks like theano functions are compiled on first use
        if compilation_lock:
            with compilation_lock:
                self.action_value.predict(np.zeros((1,) + self.observation_shape))

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
        equation = not_terminals
        equation *= np.max(freeze_q_vals, axis=1)
        equation *= discount
        targets[self.unroll, actions] = np.clip(rewards, -1., 1.) + equation
        # -----
        loss = self.action_value_train.train_on_batch(last_observations, targets)
        self.losses.append(loss)
        self.q_values.append(np.max(targets))
        print '\rIter: %8d; Loss: %9.6f; Min: %9.6f; Max: %9.6f; Avg: %9.6f --- Q-value; Min: %9.6f; Max: %9.6f; Avg: %9.6f' % (
            self.counter, loss, min(self.losses), max(self.losses), np.mean(self.losses), np.min(self.q_values),
            np.max(self.q_values), np.mean(self.q_values)),
        self.weight_switch -= 1
        if self.weight_switch < 0:
            self.weight_switch = self.switch_rate
            self.action_value.set_weights(self.action_value_train.get_weights())
            return True
        return False


def learn_proc(action_space, mem_queue, weight_dict, lock, save_rate=2000, learning_rate=0.0001, batch_size=64,
               checkpoint=0, switch_rate=200):
    import os
    pid = os.getpid()
    # -----
    agent = LearningAgent(action_space, replay_size=batch_size, compilation_lock=lock, switch_rate=switch_rate)
    # -----
    if checkpoint > 0:
        agent.action_value_train.load_weights('model-%d.h5' % (checkpoint,))
        agent.action_value.set_weights(agent.action_value_train.get_weights())
    print ' %5d> Setting weights in dict' % (pid,)
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
        counter += 1
        # -----
        last_obs[index, ...], actions[index], rewards[index], obs[index, ...], not_term[index] = mem_queue.get()
        index = (index + 1) % batch_size
        if index == 0:
            updated = agent.learn(last_obs, actions, rewards, obs, not_term, learning_rate=learning_rate)
            if updated:
                print ' %5d> Updating weights in dict' % (pid,)
                weight_dict['weights'] = agent.action_value.get_weights()
                weight_dict['update'] += 1
        # -----
        if counter % (save_rate * batch_size) == 0:
            agent.action_value_train.save_weights('model-%d.h5' % (counter,), overwrite=True)


class ActingAgent(object):
    def __init__(self, action_space, screen=(84, 84), compilation_lock=None):
        # import os
        # os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
        # from keras.models import Sequential
        # from keras.layers import InputLayer, Convolution2D, Flatten, Dense
        # from keras.optimizers import RMSprop

        self.screen = screen
        self.input_depth = 1
        self.past_range = 3
        self.observation_shape = (self.input_depth * self.past_range,) + self.screen

        self.action_value = build_network(self.observation_shape, action_space.n)
        self.action_value.compile(optimizer=RMSprop(clipnorm=1.), loss='mse')

        # Looks like theano functions are compiled on first use
        if compilation_lock:
            with compilation_lock:
                self.action_value.predict(np.zeros((1,) + self.observation_shape))

        self.action_space = action_space
        self.observations = np.zeros(self.observation_shape)
        self.last_observations = np.zeros_like(self.observations)

    def init_episode(self, observation):
        for _ in xrange(self.past_range):
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


def generate_experience_proc(start, end, epsilon, mem_queue, weight_dict, lock, game='Breakout-v0', procs=4):
    import os
    pid = os.getpid()
    # -----
    print ' %5d> Process started with %6.3f' % (pid, epsilon)
    # -----

    env = gym.make(game)
    agent = ActingAgent(env.action_space, compilation_lock=lock)

    if start > 0:
        print ' %5d> Loaded weights from file' % (pid,)
        agent.action_value.load_weights('model-%d.h5' % (start,))
    else:
        import time
        while True:
            if 'weights' in weight_dict:
                agent.action_value.set_weights(weight_dict['weights'])
                break
            time.sleep(1)
        print ' %5d> Loaded weights from dict' % (pid,)

    moves = start
    best_score = 0

    last_update = 0
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
                # mem_queue.put(agent.sars_data(action, -1, observation, False))
                break
            # -----
            mem_queue.put(agent.sars_data(action, reward, observation, not done))
            # -----
            if moves % (2000 * procs) == 0:
                print ' %5d> Epsilon: %9.6f; Best score: %4d' % (pid, decayed_epsilon, best_score)
            if moves % (50 * procs) == 0:
                update = weight_dict.get('update', 0)
                if update > last_update:
                    last_update = update
                    print ' %5d> Getting weights from dict' % (pid,)
                    agent.action_value.set_weights(weight_dict['weights'])


def main():
    manager = Manager()
    weight_dict = manager.dict()
    lock = Lock()
    mem_queue = Queue(args.queue_size)

    start = args.checkpoint
    end = args.eps_decay
    eps = [0.1, 0.05, 0.3, 0.5]

    procs = [Process(target=generate_experience_proc,
                     args=(start, end, eps[i % len(eps)], mem_queue, weight_dict, lock, args.game, args.processes))
             for i in xrange(args.processes)]

    for p in procs:
        p.start()

    # -----
    env = gym.make(args.game)
    # -----
    learn_proc(env.action_space, mem_queue, weight_dict, lock, learning_rate=args.learning_rate,
               save_rate=args.save_rate, batch_size=args.batch_size, checkpoint=start, switch_rate=args.weight_switch)

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
