import datetime
import gym_super_mario_bros
import sys
import torch

from agents.mario import Mario
from gym.wrappers import FrameStack
from metric_logger import MetricLogger
from nes_py.wrappers import JoypadSpace
from pathlib import Path
from wrappers import CustomRewardWrapper, GrayScaleObservation, ResizeObservation, SkipFrame


if __name__ == '__main__':
    # Initialize Super Mario environment (in v0.26 change render mode to 'human' to see results on the screen)
    try:
        checkpoint_step = int(sys.argv[1])
    except IndexError:
        print("Please provide the checkpoint step")
        sys.exit(1)

    try:
        render_mode = sys.argv[2]
    except IndexError:
        render_mode = 'rgb'

    try:
        reward_strategy = sys.argv[3] # points/x_pos/hybrid/legacy
    except IndexError:
        reward_strategy = 'hybrid'

    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode=render_mode, apply_api_compatibility=True)

    # Limit the action-space to
    #   0. walk right
    #   1. jump right
    # because the other actions are not useful
    env = JoypadSpace(env, [["right"], ["right", "A"]])

    env.reset()
    next_state, reward, done, trunc, info = env.step(action=0)
    print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)

    if reward_strategy != 'legacy':
        env = CustomRewardWrapper(env, reward_strategy)
        
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()

    save_dir = Path("checkpoints") / f"{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')} - {reward_strategy}"
    save_model_dir = Path("model_checkpoints") / reward_strategy
    save_dir.mkdir(parents=True)

    # get checkpoint step from sys.argv
    mario = Mario(
        state_dim=(4, 84, 84), action_dim=env.action_space.n, save_model_dir=save_model_dir, checkpoint_number=checkpoint_step)


    mario.load(f"mario_net_{checkpoint_step}_{reward_strategy}.chkpt")

    logger = MetricLogger(save_dir)

    episodes = 50_000
    for e in range(episodes):

        state = env.reset()

        # Play the game!
        while True:

            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, trunc, info = env.step(action)

            # Remember
            mario.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = mario.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            # Check if end of game
            if done or info["flag_get"]:
                break

        logger.log_episode()

        if (e % 20 == 0) or (e == episodes - 1):
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
            if mario.curr_step % mario.save_every == 0 or e == episodes - 1:
                mario.save()
