from keras.models import *
from keras.layers import *
from keras.optimizers import RMSprop
import gym
from scipy.misc import imresize
from skimage.color import rgb2gray
import numpy as np
import argparse


def build_network(input_shape, output_shape):
    state = Input(shape=input_shape)
    h = Convolution2D(16, 8, 8, subsample=(4, 4), activation='relu')(state)
    h = Convolution2D(32, 4, 4, subsample=(2, 2), activation='relu')(h)
    h = Flatten()(h)
    h = Dense(256, activation='relu')(h)

    value = Dense(1, activation='linear')(h)
    policy = Dense(output_shape, activation='softmax')(h)

    value_network = Model(input=state, output=value)
    policy_network = Model(input=state, output=policy)

    adventage = Input(shape=(1,))
    train_network = Model(input=state, output=[value, policy])

    return value_network, policy_network, train_network, adventage


class ActingAgent(object):
    def __init__(self, action_space, screen=(84, 84)):
        self.screen = screen
        self.input_depth = 1
        self.past_range = 3
        self.replay_size = 32
        self.observation_shape = (self.input_depth * self.past_range,) + self.screen

        _, self.policy, self.load_net, _ = build_network(self.observation_shape, action_space.n)

        self.load_net.compile(optimizer=RMSprop(clipnorm=1.), loss='mse')  # clipnorm=1.

        self.action_space = action_space
        self.observations = np.zeros((self.input_depth * self.past_range,) + screen)

    def init_episode(self, observation):
        for _ in range(self.past_range):
            self.save_observation(observation)

    def choose_action(self, observation):
        self.save_observation(observation)
        policy = self.policy.predict(self.observations[None, ...])[0]
        policy /= np.sum(policy)  # numpy, why?
        return np.random.choice(np.arange(self.action_space.n), p=policy)

    def save_observation(self, observation):
        self.observations = np.roll(self.observations, -self.input_depth, axis=0)
        self.observations[-self.input_depth:, ...] = self.transform_screen(observation)

    def transform_screen(self, data):
        return rgb2gray(imresize(data, self.screen))[None, ...]


parser = argparse.ArgumentParser(description='Evaluation of model')
parser.add_argument('--game', default='Breakout-v0', help='Name of openai gym environment', dest='game')
parser.add_argument('--evaldir', default=None, help='Directory to save evaluation', dest='evaldir')
parser.add_argument('--model', help='File with weights for model', dest='model')


def main():
    args = parser.parse_args()
    # -----
    env = gym.make(args.game)
    if args.evaldir:
        env.monitor.start(args.evaldir)
    # -----
    agent = ActingAgent(env.action_space)

    model_file = args.model

    agent.load_net.load_weights(model_file)

    game = 1
    for _ in range(10):
        done = False
        episode_reward = 0
        noops = 0

        # init game
        observation = env.reset()
        agent.init_episode(observation)
        # play one game
        print('Game #%8d; ' % (game,), end='')
        while not done:
            env.render()
            action = agent.choose_action(observation)
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            # ----
            if action == 0:
                noops += 1
            else:
                noops = 0
            if noops > 100:
                break
        print('Reward %4d; ' % (episode_reward,))
        game += 1
    # -----
    if args.evaldir:
        env.monitor.close()


if __name__ == "__main__":
    main()
