from connectFour3D import ConnectFour3D, Player
from player_functions import *
from typing import Tuple
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
import numpy as np
from gymnasium import spaces



CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

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


# class Env:
#     def __init__(self):
#         ...

#     def step() -> Tuple[step, reward, done, info]:
#         # Step = place piece,
#         # reward = reward function for each step,
#         # done == Game over, done,
#         # info = dict (State of game, player turn, pieces placed, etc.)
#         ...


class ConnectFour3DEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, connect_four, players):
        super().__init__()
        self.action_space = spaces.Box(low=0, high=15) # np.array(range(16)) This should possible change per turn
        self.observation_space = spaces.Box(low=0, high=2,
                                            shape=(4, 4, 4), dtype=np.uint8)
        self.connect_four = connect_four
        self.players = players
        self.connect_four.play(players=players)

    def step(self, action): # -> Tuple[step, reward, done, info]:
        ...
        return observation, reward, done, info

    def reset(self):
        self.connect_four.__init__()
        observation = self.connect_four.board
        return observation

    def render(self):
        ...

    def close(self):
        ...



if __name__ == "__main__":
    players = [
        Player(name="A", function=place_random, piece_value=1),
        Player(name="B", function=place_random, piece_value=2),
    ]

    connect_four = ConnectFour3D()
    # connect_four.play(players=players, display=True)
    env = ConnectFour3DEnv(connect_four, players) # TODO create
    callback = TrainAndLoggingCallback(check_freq=10_000, save_path=CHECKPOINT_DIR)
    model = PPO(policy='MlpPolicy', env=env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)
    model.learn(total_timesteps=10, callback=callback) # 1_000_000
    
