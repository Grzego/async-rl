import gym
from scipy.misc import imresize
from skimage.color import rgb2gray
import numpy as np
import argparse


def build_network(input_shape, output_shape):
    from keras.models import Model
    from keras.layers import Input, Conv2D, Flatten, Dense

    x = Input(shape=input_shape)
    h = Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation='relu', data_format='channels_first')(x)
    h = Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', data_format='channels_first')(h)
    h = Flatten()(h)
    h = Dense(256, activation='relu')(h)
    v = Dense(output_shape, activation='linear')(h)
    return Model(inputs=x, outputs=v)


class ActingAgent(object):
    def __init__(self, action_space, screen=(84, 84)):
        from keras.optimizers import RMSprop

        self.screen = screen
        self.input_depth = 1
        self.past_range = 3
        self.replay_size = 32
        self.observation_shape = (self.input_depth * self.past_range,) + self.screen

        self.action_value = build_network(self.observation_shape, action_space.n)
        self.action_value.compile(optimizer=RMSprop(clipnorm=1.), loss='mse')  # clipnorm=1.

        self.action_space = action_space
        self.observations = np.zeros((self.input_depth * self.past_range,) + screen)

    def init_episode(self, observation):
        for _ in range(self.past_range):
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
        print('Reward %4d; ' % (episode_reward,))
        game += 1
    # -----
    if args.evaldir:
        env.monitor.close()


if __name__ == "__main__":
    main()
