import numpy as np
from typing import List


def place_random(board: np.array, available_slots: List[int]):
    return np.random.choice(available_slots)
