import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import config

from wrappers import SparseRewardWrapper


def evaluate(model_path):

    env = gym.make("CartPole-v1")
    env = SparseRewardWrapper(env, shaping=False)

    model = PPO.load(model_path)

    wins = 0
    rewards = []

    for _ in range(config.EVAL_EPISODES):

        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)

        if total_reward >= 500:  # CartPole solved threshold
            wins += 1

    print("Win Rate:", wins / config.EVAL_EPISODES)
    print("Average Reward:", np.mean(rewards))
    print("Reward Std:", np.std(rewards))

    env.close()


if __name__ == "__main__":
    evaluate("outputs/models/ppo_cartpole")