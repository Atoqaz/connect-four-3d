import numpy as np
from typing import List
from stable_baselines3 import PPO
from pathlib import Path

DIR = Path(__file__).parent
models = []
for p in DIR.joinpath("train").glob("*.zip"):
    models.append(int(str(p).split("\\")[-1].split(".")[0]))
if models:
    HIGHEST_MODEL = PPO.load(f"./train/{max(models)}")
else:
    HIGHEST_MODEL = None


def place_random(board: np.array, available_slots: List[int]):
    return np.random.choice(available_slots)


def place_minimum(board: np.array, available_slots: List[int]):
    return min(available_slots)


def rl_model_highest(board: np.array, available_slots: List[int]):
    def switch(
        board, a=1, b=2
    ):  # Model is trained on 1 being itself, and 2 is opponent
        board = np.where(board == a, np.inf, board)
        board = np.where(board == b, a, board)
        board = np.where(board == np.inf, b, board)
        return board

    def get_highest_model():
        models = []
        for p in DIR.joinpath("train").glob("*.zip"):
            models.append(int(str(p).split("\\")[-1].split(".")[0]))
        if models:
            highest_model = PPO.load(f"./train/{max(models)}")
            # print(f"Model: {max(models)}.zip")
        else:
            highest_model = None
        return highest_model

    model = get_highest_model()
    action = None
    while action not in available_slots:
        if model is None:
            action = place_random(board=board, available_slots=available_slots)
        else:
            action, state = model.predict(switch(board))
            action = int(action)
    return action
