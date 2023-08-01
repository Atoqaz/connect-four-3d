# %%

from tictactoe import TicTacToe, Player
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


class TicTacToeRL(TicTacToe):
    def start_game(self):
        self._create_board()
        self.turn = 0

    def _reward_max_min(self, board):
        # board index
        diag_1 = np.array([0, 4, 8])  # diagonal top left to bottom right
        diag_2 = np.array([2, 4, 6])  # diagonal top right to bottom left

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

        for i in range(self.width):
            player_sum, opponent_sum = _row_count(
                row=board[i, :],  # Rows
                player_sum=player_sum,
                opponent_sum=opponent_sum,
            )
            player_sum, opponent_sum = _row_count(
                row=board[:, i],  # Columns
                player_sum=player_sum,
                opponent_sum=opponent_sum,
            )

        player_sum, opponent_sum = _row_count(
            row=board.ravel()[diag_1],  # Rows
            player_sum=player_sum,
            opponent_sum=opponent_sum,
        )
        player_sum, opponent_sum = _row_count(
            row=board.ravel()[diag_2],  # Rows
            player_sum=player_sum,
            opponent_sum=opponent_sum,
        )
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


class TicTacToeEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, players):
        super().__init__()
        self.action_space = Discrete(9)
        print(f"self.action_space: {type(self.action_space)}")
        self.observation_space = Box(low=0, high=2, shape=(3, 3))  # , dtype=np.uint8
        self.tic_tac_toe = TicTacToeRL()
        self.players = players
        self.tic_tac_toe.start_game()

    def step(self, action) -> Tuple[np.array, int, bool, Dict[str, str]]:
        observation, reward, done, info = self.tic_tac_toe.make_move_step(
            piece_placement=int(action), players=self.players
        )
        return observation, reward, done, info

    def reset(self):
        self.tic_tac_toe.start_game()
        observation = self.tic_tac_toe.board
        return observation

    def render(self):
        print(self.tic_tac_toe.board)

    def close(self):
        ...


if __name__ == "__main__":
    check_freq = 10_000  # 10_000
    steps = 100_000  # 1_000_000
    players = [
        Player(name="A", function=None, piece_value=1),  # RL model
        Player(name="B", function=rl_model_highest, piece_value=2),
    ]
    env = TicTacToeEnv(players)
    callback = TrainAndLoggingCallback(check_freq=check_freq, save_path=CHECKPOINT_DIR)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=2,
        tensorboard_log=LOG_DIR,
        learning_rate=0.000001,  # 0.000001
        n_steps=512,  # 512
    )
    model.learn(total_timesteps=steps, callback=callback)


# %%
