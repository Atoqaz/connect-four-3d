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


class ConnectFour3DRL(ConnectFour3D):
    def start_game(self):
        self._create_board()
        self.turn = 0

    def _reward_max_min(self, board):
        flat_diag1 = np.array([0, 5, 10, 15])  # index for the layer
        flat_diag2 = np.array([3, 6, 9, 12])  # index for the layer
        diag_x1 = np.array([0, 17, 34, 51])  # index for the board
        diag_x2 = np.array([3, 18, 33, 48])  # index for the board
        diag_y1 = np.array([0, 20, 40, 60])  # index for the board
        diag_y2 = np.array([12, 24, 36, 48])  # index for the board
        diag_x_offset = 4
        diag_y_offset = 1
        diag_3d = np.array(
            [[0, 21, 42, 63], [15, 26, 37, 48], [3, 22, 41, 60], [12, 25, 38, 51]]
        )
        # unique, counts = np.unique(board, return_counts=True)

        def _row_count(
            row: np.array,
            player_sum: float = 0,
            opponent_sum: float = 0,
        ) -> Tuple[float, float]:
            unique, counts = np.unique(row, return_counts=True)
            player_val, opponent_val = 0, 0
            for u, c in zip(unique, counts):
                if u == 1:
                    player_val = c
                elif u == 2:
                    opponent_val = c
            player_sum += player_val
            opponent_sum += opponent_val
            return player_sum, opponent_sum

        player_sum = 0
        opponent_sum = 0

        for layer in board:
            for i in range(4):
                player_sum, opponent_sum = _row_count(
                    row=layer[i, :],
                    player_sum=player_sum,
                    opponent_sum=opponent_sum,
                )  # Every row in X (16)
                player_sum, opponent_sum = _row_count(
                    row=layer[:, i],
                    player_sum=player_sum,
                    opponent_sum=opponent_sum,
                )  # Every row in Y (16)

            player_sum, opponent_sum = _row_count(
                row=layer.ravel()[flat_diag1],
                player_sum=player_sum,
                opponent_sum=opponent_sum,
            )  # Flat diagonals (8) [1/2]
            player_sum, opponent_sum = _row_count(
                row=layer.ravel()[flat_diag2],
                player_sum=player_sum,
                opponent_sum=opponent_sum,
            )  # Flat diagonals (8) [2/2]
        for i in range(4):
            for j in range(4):
                player_sum, opponent_sum = _row_count(
                    row=board[:, i, j],
                    player_sum=player_sum,
                    opponent_sum=opponent_sum,
                )  # Column (16)

            player_sum, opponent_sum = _row_count(
                row=board[:, i, j],
                player_sum=player_sum,
                opponent_sum=opponent_sum,
            )  # Column (16)
            player_sum, opponent_sum = _row_count(
                row=board.ravel()[diag_x1 + diag_x_offset * i],
                player_sum=player_sum,
                opponent_sum=opponent_sum,
            )  # Row bottom to top diagonals in X (8) [1/2]
            player_sum, opponent_sum = _row_count(
                row=board.ravel()[diag_x2 + diag_x_offset * i],
                player_sum=player_sum,
                opponent_sum=opponent_sum,
            )  # Row bottom to top diagonals in X (8) [2/2]
            player_sum, opponent_sum = _row_count(
                row=board.ravel()[diag_y1 + diag_y_offset * i],
                player_sum=player_sum,
                opponent_sum=opponent_sum,
            )  # Row bottom to top diagonals in Y (8) [1/2]
            player_sum, opponent_sum = _row_count(
                row=board.ravel()[diag_y2 + diag_y_offset * i],
                player_sum=player_sum,
                opponent_sum=opponent_sum,
            )  # Row bottom to top diagonals in Y (8) [2/2]

        for diag in diag_3d:
            player_sum, opponent_sum = _row_count(
                row=board.ravel()[diag],
                player_sum=player_sum,
                opponent_sum=opponent_sum,
            )  # 3D Diagonals (4)
        return player_sum, opponent_sum

    def reward(self, board: np.array, piece_placement: int):
        player_sum, opponent_sum = self._reward_max_min(board=board)
        return player_sum - opponent_sum

    def make_move_step(self, piece_placement, players, display=False):
        available_slots = self.get_available_slots(board=self.board)

        if available_slots:
            player = players[self.turn]
            if piece_placement in available_slots:
                # Move for AI
                self.board = self.make_move(
                    board=self.board, player=player, piece_placement=piece_placement
                )
                if self._detect_win(board=self.board):
                    if display:
                        print(f"Game won by player: {player.name}")
                    return (
                        self.board,
                        1000,
                        True,
                        {},
                    )  # (observation, reward, done, info) AI won
                self.turn = (self.turn + 1) % 2  # Next player in turn

                # Move for opponent
                player = players[self.turn]
                available_slots = self.get_available_slots(board=self.board)
                piece_placement = player.function(
                    board=self.board,
                    available_slots=available_slots,
                )
                self.board = self.make_move(
                    board=self.board, player=player, piece_placement=piece_placement
                )
                if self._detect_win(board=self.board):
                    if display:
                        print(f"Game won by player: {player.name}")
                    return (
                        self.board,
                        -1000,
                        True,
                        {},
                    )  # (observation, reward, done, info) Opponent won
                self.turn = (self.turn + 1) % 2  # Next player in turn

                reward = self.reward(self.board, piece_placement)
                return (
                    self.board,
                    reward,
                    False,
                    {},
                )  # (observation, reward, done, info) Opponent won

            else:  # Not a valid move
                return self.board, -1, 0, {}
        else:  # Game tied
            return self.board, 1, 1, {}  # "Tied"


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
            model_path = os.path.join(self.save_path, f"{self.n_calls}")
            self.model.save(model_path)
        return True


class ConnectFour3DEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, players):
        super().__init__()
        self.action_space = Discrete(16)
        print(f"self.action_space: {type(self.action_space)}")
        self.observation_space = Box(low=0, high=2, shape=(4, 4, 4))  # , dtype=np.uint8
        self.connect_four = ConnectFour3DRL()
        self.players = players
        self.connect_four.start_game()

    def step(self, action) -> Tuple[np.array, int, bool, Dict[str, str]]:
        observation, reward, done, info = self.connect_four.make_move_step(
            piece_placement=int(action), players=self.players
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


class TicTacToeEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, players):
        super().__init__()
        self.action_space = Discrete(9)
        print(f"self.action_space: {type(self.action_space)}")
        self.observation_space = Box(low=0, high=2, shape=(3,3))  # , dtype=np.uint8
        self.connect_four = ConnectFour3DRL()
        self.players = players
        self.connect_four.start_game()

    def step(self, action) -> Tuple[np.array, int, bool, Dict[str, str]]:
        observation, reward, done, info = self.connect_four.make_move_step(
            piece_placement=int(action), players=self.players
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
    check_freq = 10_000  # 10_000
    steps = 100_000  # 1_000_000
    players = [
        Player(name="A", function=None, piece_value=1),  # RL model
        Player(name="B", function=rl_model_highest, piece_value=2),
    ]
    env = ConnectFour3DEnv(players)
    callback = TrainAndLoggingCallback(
        check_freq=check_freq, save_path=CHECKPOINT_DIR
    )
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=2,
        tensorboard_log=LOG_DIR,
        learning_rate=0.000001, # 0.000001
        n_steps=512, # 512
    )
    model.learn(total_timesteps=steps, callback=callback) 


# %%
