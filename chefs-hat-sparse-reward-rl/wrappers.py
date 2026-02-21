import gymnasium as gym


class SparseRewardWrapper(gym.Wrapper):
    """
    Sparse vs Reward Shaping wrapper.
    Variant: ID % 7 = 3
    """

    def __init__(self, env, shaping=True):
        super().__init__(env)
        self.shaping = shaping
        self.prev_cards = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_cards = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaping_reward = 0

        if self.shaping:

            # Intermediate card reduction reward
            if "cards_left" in info:
                current_cards = info["cards_left"]

                if self.prev_cards is not None:
                    shaping_reward += 0.05 * (self.prev_cards - current_cards)

                self.prev_cards = current_cards

            # Ranking reward
            if "rank" in info:
                shaping_reward += 0.1 * (4 - info["rank"])

        total_reward = reward + shaping_reward

        return obs, total_reward, terminated, truncated, info
