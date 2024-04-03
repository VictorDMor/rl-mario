import numpy as np

# Initialize Q-table
Q = {}

# Parameters
epsilon = 0.9  # Exploration rate
learning_rate = 0.7  # Learning rate
discount_factor = 0.9  # Discount factor
num_episodes = 5000  # Number of episodes for training

# Q-Learning
for i_episode in range(num_episodes):
    state = env.reset()
    x, y = get_mario_position(state)

    for t in range(100):
        # Choose action
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = max(Q[(x, y)], key=Q[(x, y)].get)  # Exploit

        # Take action
        state, reward, done, info = env.step(action)
        new_x, new_y = get_mario_position(state)

        # Update Q-table
        if (x, y) not in Q:
            Q[(x, y)] = {action: 0 for action in range(env.action_space.n)}
        if (new_x, new_y) not in Q:
            Q[(new_x, new_y)] = {
                action: 0 for action in range(env.action_space.n)}
        Q[(x, y)][action] = Q[(x, y)][action] + learning_rate * (reward +
                                                                 discount_factor * max(Q[(new_x, new_y)].values()) - Q[(x, y)][action])

        # Update position
        x, y = new_x, new_y

        if done:
            break

    # Decrease epsilon
    epsilon *= 0.99
