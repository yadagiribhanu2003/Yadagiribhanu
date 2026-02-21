from stable_baselines3 import PPO
import config


def create_agent(env):
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.LEARNING_RATE,
        gamma=config.GAMMA,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        ent_coef=config.ENT_COEF,
        clip_range=config.CLIP_RANGE,
        verbose=1,
        tensorboard_log=config.LOG_DIR,
        seed=config.SEED
    )
    return model
