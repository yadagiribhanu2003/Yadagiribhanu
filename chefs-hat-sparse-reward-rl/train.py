import torch
import numpy as np
import os
import config
import gymnasium as gym
from wrappers import SparseRewardWrapper
from agents.ppo_agent import create_agent
from callbacks import WinRateCallback
from utils import plot_results

def make_env(shaping=True):

    env = gym.make("CartPole-v1")
    env = SparseRewardWrapper(env, shaping=shaping)
    return env


if __name__ == "__main__":


    # Create Directories
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.CSV_DIR, exist_ok=True)
    os.makedirs(config.PLOT_DIR, exist_ok=True)

    # ======================
    # Set Seeds (Reproducibility)
    # ======================
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # ======================
    # Choose Variant
    # ======================
    shaping = True

    print(f"\n Training started | Shaping = {shaping}\n")

    # Create Environments

    env = make_env(shaping=shaping)
    eval_env = make_env(shaping=False)


    # Create Model

    model = create_agent(env)


    callback = WinRateCallback(
        eval_env,
        eval_episodes=config.EVAL_EPISODES,
        save_name="live_results.csv",
        verbose=1
    )

    # Train Model
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=callback,
        progress_bar=True
    )

    # Save Model
    model_name = "ppo_shaping" if shaping else "ppo_sparse"
    model_path = os.path.join(config.MODEL_DIR, model_name)

    model.save(model_path)

    print(f"\n Model saved at: {model_path}.zip")

    # Save Final CSV
    csv_filename = f"{model_name}_results.csv"
    csv_path = os.path.join(config.CSV_DIR, csv_filename)

    callback.save_results(csv_filename)

    print(f" CSV saved at: {csv_path}")

    # Plot Results
    try:
        plot_name = f"{model_name}.png"
        plot_results(csv_path, plot_name)

        print(f" Plot saved at: {os.path.join(config.PLOT_DIR, plot_name)}")

    except Exception as e:
        print(" Plotting failed:", e)

    # Cleanup
    env.close()
    eval_env.close()

    print("\n Training Complete!\n")