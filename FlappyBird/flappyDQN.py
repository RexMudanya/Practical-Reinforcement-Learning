import random
import numpy as np
import flappy_bird_gym
from collections import deque
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import load_model, save_model, Sequential
from tensorflow.keras.optimizers import RMSprop


# NN for agent
def NeuralNetwork(input_shape, output_shape):
    model = Sequential()

    model.add(Dense(512, input_shape=input_shape, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(output_shape, activation='linear', kernel_initializer='he_uniform'))

    model.compile(loss='mse', optimizer=RMSprop(lr=0.0001, rho=0.95, epsilon=0.01), metrics=['accuracy'])

    model.summary()

    return model


class DQNAgent:
    def __init__(self):
        # environment variables
        self.env = flappy_bird_gym.make("FlappyBird-v0")
        self.episodes = 1000
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.memory = deque(maxlen=2000)

        # hyperparams
        self.gamma = 0.95  # discount rate
        self.epsilon = 1  # probability of taking a random action
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01  # 1% chance of random action
        self.batch_number = 64  # 16, 32, 128, 256

        self.train_start = 1000
        self.jump_prob = 0.01

        self.model = NeuralNetwork((self.state_space, ), self.action_space)


if __name__ == '__main__':
    agent = DQNAgent()
