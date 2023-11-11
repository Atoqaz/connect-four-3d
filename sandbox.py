# # from pathlib import Path

# # DIR = Path(__file__).parent

# # models = []
# # for p in DIR.joinpath("train").glob("*.zip"):
# #     models.append(int(str(p).split("\\")[-1].split(".")[0]))

# # print(max(models))

import numpy as np

# from typing import Tuple, Dict
# from tictactoe import TicTacToe, Player


# players = [
#     Player(name="A", function=None, piece_value=1),
#     Player(name="B", function=None, piece_value=2),
# ]

# tic_tac_toe = TicTacToe()
# tic_tac_toe.play(players=players, display=True)


scoreboard = np.zeros((9, 2))
scoreboard[0, :] = (0, 3)
scoreboard[5, :] = (5, 8)
scoreboard[2, :] = (2, -2)
scoreboard[3, :] = (3, 4)

scoreboard2 = np.copy(scoreboard)
# print(scoreboard2)
scoreboard2[5,:] = (999, 999)

print(scoreboard)
print(scoreboard2)
# print(scoreboard[:, 1])
# print(np.argmax(scoreboard[:, 1]))
# print(scoreboard[np.argmax(scoreboard[:, 1]), 0])
