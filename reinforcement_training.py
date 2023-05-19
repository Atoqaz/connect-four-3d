# %%
from connectFour3D import ConnectFour3D, Player
from player_functions import *
from typing import Tuple, Dict
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym

from gym.spaces.discrete import Discrete
from gym.spaces import Box

# import torch as th


CHECKPOINT_DIR = "./train/"
LOG_DIR = "./logs/"


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f"best_model_{self.n_calls}")
            self.model.save(model_path)
        return True


class ConnectFour3DEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, players):
        super().__init__()
        self.action_space = Discrete(
            16
        )
        print(f"self.action_space: {type(self.action_space)}")
        self.observation_space = Box(low=0, high=2, shape=(4, 4, 4))  # , dtype=np.uint8
        self.connect_four = ConnectFour3D()
        self.players = players
        self.connect_four.start_game()

    def step(self, action) -> Tuple[np.array, int, bool, Dict[str, str]]:
        observation, reward, done, info = self.connect_four.make_move_step(
            action, self.players
        )
        return observation, reward, done, info

    def reset(self):
        self.connect_four.start_game()
        observation = self.connect_four.board
        return observation

    def render(self):
        print(self.connect_four.board)

    def close(self):
        ...


if __name__ == "__main__":
    players = [
        Player(name="A", function=None, piece_value=1),
        Player(name="B", function=place_random, piece_value=2),
    ]

    env = ConnectFour3DEnv(players)
    callback = TrainAndLoggingCallback(check_freq=10_000, save_path=CHECKPOINT_DIR)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=2,
        tensorboard_log=LOG_DIR,
        learning_rate=0.000001,
        n_steps=512,
    )
    # model.to("cuda")
    model.learn(total_timesteps=100_000, callback=callback)  # 1_000_000


# %%
