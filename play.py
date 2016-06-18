import gym
from scipy.misc import imresize
from skimage.color import rgb2gray
import numpy as np
import argparse


class ActingAgent(object):
    def __init__(self, action_space, screen=(84, 84), compilation_lock=None):
        from keras.models import Sequential
        from keras.layers import InputLayer, Convolution2D, Flatten, Dense
        from keras.optimizers import RMSprop

        self.screen = screen
        self.input_depth = 1
        self.past_range = 3
        self.replay_size = 32
        self.observation_shape = (self.input_depth * self.past_range,) + self.screen

        self.action_value = Sequential([
            InputLayer(input_shape=self.observation_shape),
            Convolution2D(16, 8, 8, subsample=(4, 4), activation='relu'),
            Convolution2D(32, 4, 4, subsample=(2, 2), activation='relu'),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(action_space.n, activation='linear'),
        ])

        if compilation_lock:
            with compilation_lock:
                self.action_value.compile(optimizer=RMSprop(clipnorm=1.), loss='mse')  # clipnorm=1.
        else:
            self.action_value.compile(optimizer=RMSprop(clipnorm=1.), loss='mse')  # clipnorm=1.

        self.action_space = action_space
        self.observations = np.zeros((self.input_depth * self.past_range,) + screen)

    def init_episode(self, observation):
        for _ in xrange(self.past_range):
            self.save_observation(observation)

    def choose_action(self, observation, epsilon=0.0):
        self.save_observation(observation)
        if np.random.random() < epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.action_value.predict(self.observations[None, ...]))

    def save_observation(self, observation):
        self.observations = np.roll(self.observations, -self.input_depth, axis=0)
        self.observations[-self.input_depth:, ...] = self.transform_screen(observation)

    def transform_screen(self, data):
        return rgb2gray(imresize(data, self.screen))[None, ...]


parser = argparse.ArgumentParser(description='Evaluation of model')
parser.add_argument('--game', default='Breakout-v0', help='Name of openai gym environment', dest='game')
parser.add_argument('--evaldir', default=None, help='Directory to save evaluation', dest='evaldir')
parser.add_argument('--model', help='File with weights for model', dest='model')
parser.add_argument('--eps', default=0., help='Epsilon value', dest='eps', type=float)


def main():
    args = parser.parse_args()
    # -----
    env = gym.make(args.game)
    if args.evaldir:
        env.monitor.start(args.evaldir)
    # -----
    agent = ActingAgent(env.action_space)

    model_file = args.model
    epsilon = args.eps

    agent.action_value.load_weights(model_file)

    game = 1
    for _ in xrange(10):
        done = False
        episode_reward = 0
        noops = 0

        # init game
        observation = env.reset()
        agent.init_episode(observation)
        # play one game
        print 'Game #%8d; ' % (game,),
        while not done:
            env.render()
            action = agent.choose_action(observation, epsilon=epsilon)
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            # ----
            if action == 0:
                noops += 1
            else:
                noops = 0
            if noops > 100:
                break
        print 'Reward %4d; ' % (episode_reward,)
        game += 1
    # -----
    if args.evaldir:
        env.monitor.close()


if __name__ == "__main__":
    main()
