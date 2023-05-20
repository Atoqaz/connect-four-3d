# from pathlib import Path

# DIR = Path(__file__).parent

# models = []
# for p in DIR.joinpath("train").glob("*.zip"):
#     models.append(int(str(p).split("\\")[-1].split(".")[0]))

# print(max(models))

import numpy as np
from typing import Tuple, Dict


def row_count(
    row: np.array, player_weight: float = 1, opponent_weight: float = 1
) -> Tuple[float, float]:
    unique, counts = np.unique(row, return_counts=True)
    p_val, o_val = 0, 0
    for u, c in zip(unique, counts):
        if u == 1:
            p_val = c * player_weight
        elif u == 2:
            o_val = c * opponent_weight
    return p_val, o_val


board = np.ones(5)

board[0] = 0
# board[2] = 2

# print(board)
# unique, counts = np.unique(board, return_counts=True)
unique, counts = row_count(board, player_weight=10, opponent_weight=5)
print(unique)
print(counts)
