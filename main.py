import gym_super_mario_bros
import numpy as np
import pickle

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation

if __name__ == "__main__":
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)

    # Initialize Q-table
    try:
        with open('q_table.pkl', 'rb') as f:
            Q = pickle.load(f)
    except FileNotFoundError:
        Q = {}  # Initialize an empty Q-table

    # Parameters
    epsilon = 0.9  # Exploration rate
    learning_rate = 0.7  # Learning rate
    discount_factor = 0.9  # Discount factor
    num_episodes = 5000  # Number of episodes for training
    info = {}

    # Q-Learning
    for i_episode in range(num_episodes):
        state = env.reset()

        while True:
            # Choose action
            x, y = info.get('x_pos', 0), info.get('y_pos', 0)

            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = max(Q[(x, y)], key=Q[(x, y)].get)  # Exploit

            # Take action
            state, reward, done, info = env.step(action)
            new_x, new_y = info.get('x_pos', 0), info.get('y_pos', 0)
            env.render()

            # Update Q-table
            if (x, y) not in Q:
                Q[(x, y)] = {action: np.random.uniform(low=0, high=0.01) for action in range(env.action_space.n)}
            if (new_x, new_y) not in Q:
                Q[(new_x, new_y)] = {action: np.random.uniform(low=0, high=0.01) for action in range(env.action_space.n)}
            Q[(x, y)][action] = Q[(x, y)][action] + learning_rate * (reward +
                                                                    discount_factor * max(Q[(new_x, new_y)].values()) - Q[(x, y)][action])

            # Update position
            x, y = new_x, new_y

            if done:
                break

        # Decrease epsilon
        epsilon *= 0.99

    with open('q_table.pkl', 'wb') as f:
        pickle.dump(Q, f)
