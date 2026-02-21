import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
import config
import os


class WinRateCallback(BaseCallback):
    """
    Custom callback to evaluate model performance during training.
    Logs:
    - Win rate
    - Average reward
    - Timesteps
    """

    def __init__(self, eval_env, eval_episodes=10, save_name="live_results.csv", verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_episodes = eval_episodes
        self.save_name = save_name

        # Storage
        self.timesteps = []
        self.win_rates = []
        self.avg_rewards = []

        # Create directory once
        os.makedirs(config.CSV_DIR, exist_ok=True)

    # REQUIRED by Stable-Baselines3
    def _on_step(self) -> bool:
        return True

    # Runs after each rollout
    def _on_rollout_end(self):

        wins = 0
        rewards = []

        for _ in range(self.eval_episodes):

            obs, info = self.eval_env.reset()
            done = False
            total_reward = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                total_reward += reward

            rewards.append(total_reward)

            # CartPole solved threshold
            if total_reward >= 500:
                wins += 1

        # Metrics
        win_rate = wins / self.eval_episodes
        avg_reward = np.mean(rewards)

        # Store results
        self.timesteps.append(self.num_timesteps)
        self.win_rates.append(win_rate)
        self.avg_rewards.append(avg_reward)

        # Print progress
        if self.verbose > 0:
            print(
                f"Timestep: {self.num_timesteps} | "
                f"Win Rate: {win_rate:.2f} | "
                f"Avg Reward: {avg_reward:.2f}"
            )

        # ✅ Save results after each rollout
        self._save_to_csv()

    def _save_to_csv(self):
        """
        Internal method to save results continuously
        """
        path = os.path.join(config.CSV_DIR, self.save_name)

        df = pd.DataFrame({
            "timesteps": self.timesteps,
            "win_rate": self.win_rates,
            "avg_reward": self.avg_rewards
        })

        df.to_csv(path, index=False)

        if self.verbose > 0:
            print(f"CSV updated: {path}")

    def save_results(self, filename=None):
        """
        Optional manual save (e.g., at end of training)
        """
        if filename is None:
            filename = self.save_name

        path = os.path.join(config.CSV_DIR, filename)

        df = pd.DataFrame({
            "timesteps": self.timesteps,
            "win_rate": self.win_rates,
            "avg_reward": self.avg_rewards
        })

        df.to_csv(path, index=False)

        if self.verbose > 0:
            print(f"Final CSV saved: {path}")