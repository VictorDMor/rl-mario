import gym_super_mario_bros
import matplotlib.pyplot as plt
import numpy as np
import os

from agents.dqn import DQNAgent
from datetime import datetime
from nes_py.wrappers import JoypadSpace
from gym import ObservationWrapper
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from tensorflow.keras.models import load_model

class CustomObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super(CustomObservationWrapper, self).__init__(env)
    
    def observation(self, obs):
        # Preprocess the observation (state)
        return preprocess_observation(obs)

def test_preprocess(current):
    # Plot the original and preprocessed images side by side
    plt.imshow(current)
    plt.title('Preprocessed Observation with Custom Wrapper')
    plt.axis('off')
    plt.show()

def preprocess_observation(obs):
    # Crop the screen (remove the status bar at the top and the 8 last columns)
    cropped_obs = obs[48:-8, 8:-8]
    # Reshape the observation to have a channel dimension
    processed_obs = cropped_obs.reshape(cropped_obs.shape[0], cropped_obs.shape[1], 1)
    return processed_obs

if __name__ == '__main__':

    episode_rewards = []
    
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env)
    env = CustomObservationWrapper(env)

    # Create checkpoint directory
    checkpoint_dir = './checkpoints'  # Directory to save checkpoints
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Display the initial environment (first frame)
    obs = env.reset()
    # test_preprocess(obs)
    env.render()

    agent = DQNAgent((128, 224, 1), env.action_space.n)

    checkpoint = './checkpoints/mario_model_40.h5'

    try:
        agent.model = load_model(checkpoint)
        print(f"Loaded checkpoint: {checkpoint}")
    except Exception as e:
        print(f"Failed to load checkpoint: {checkpoint}")
        print(e)

    num_episodes = 100  # This is just an example value
    batch_size = 32  # The batch size for experience replay

    for e in range(num_episodes):
        # Reset the environment for a new episode
        state = preprocess_observation(env.reset())
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        done = False
        total_reward = 0

        while not done:
            # Act according to the current policy
            env.render()
            action = agent.act(state)
            
            # Take the action and observe the next state and reward
            next_state, reward, done, info = env.step(action)
            next_state = preprocess_observation(next_state)
            next_state = np.expand_dims(next_state, axis=0)
            
            # Remember the experience
            agent.remember(state, action, reward, next_state, done)
            
            # Move to the next state
            state = next_state
            total_reward += reward

            # If done, exit the loop
            if done:
                print(f"Episode: {e}/{num_episodes}, Total Reward: {total_reward}")
                episode_rewards.append(total_reward)
                # Inside your training loop, at the end of an episode
                if e % 10 == 0:  # For example, save every 10 episodes
                    # get current datetime to string
                    now = datetime.now().strftime("%Y%m%d%H%M%S")
                    checkpoint_path = os.path.join(checkpoint_dir, f'mario_model_{now}.h5')
                    agent.model.save(checkpoint_path)
                    print(f"Checkpoint saved to {checkpoint_path}")
                break

        # Train the agent with the experiences
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    plt.plot(episode_rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()