import numpy as np
from typing import List
from stable_baselines3 import PPO

MODEL = PPO.load("./train/best_model_10000")


def place_random(board: np.array, available_slots: List[int]):
    return np.random.choice(available_slots)


def place_minimum(board: np.array, available_slots: List[int]):
    return min(available_slots)


def rl_model(board: np.array, available_slots: List[int]):
    model = MODEL
    action = None
    while action not in available_slots:
        action, state = model.predict(board)
    return action
