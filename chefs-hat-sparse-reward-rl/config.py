import os

# ======================
# Environment
# ======================
ENV_NAME = "CartPole-v1"

SEED = 42
TOTAL_TIMESTEPS = 100_000

# ======================
# PPO Hyperparameters
# ======================
LEARNING_RATE = 3e-4
GAMMA = 0.99
N_STEPS = 1024
BATCH_SIZE = 64
ENT_COEF = 0.0
CLIP_RANGE = 0.2

EVAL_EPISODES = 5

# ======================
# Paths
# ======================
BASE_DIR = "outputs"
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
CSV_DIR = os.path.join(BASE_DIR, "csv")
