"""

Create Board (done)
Get available slots 

Make move (isPlayerOneTurn: bool)

Create winning function

Display board

"""

#%%
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass


@dataclass()
class Player:
    name: str  # Unique
    function: callable
    piece_value: int = None

    def __repr__(self):
        return f"Player(Name = {self.name})"


class ConnectFour3D:
    """Players: 2 players with values 1 & 2. 0 is empty slot.
    """

    def __init__(self):
        self.board = self._create_board()

    def _create_board(self) -> np.array:
        return np.zeros([4, 4, 4], dtype=int)

    def _get_available_slots(self, board: np.array) -> np.array:
        return (board[-1] == 0).astype(int).ravel()

    def make_move(
        self, board: np.array, player: Player, piece_placement: int
    ) -> np.array:
        available_slots = self._get_available_slots(board=self.board)
        if not available_slots[piece_placement]:
            raise ValueError(
                f"The selected piece placement is not available. Chosen placement: {piece_placement}"
            )

        for layer in board:
            location = layer.ravel()[piece_placement]
            if location == 0:
                location = player.piece_value
                return board

    def _next_player(self, player: Player):
        for index, _player in enumerate(self.players):
            if _player.name == player.name:
                break
        if index == len(self.players) - 1:
            player = self.players[0]
        else:
            player = self.players[index + 1]
        return player

    def _check_identical(self, list: np.array) -> bool:
        return (list == list[0]).all()

    def _detect_win(self, board: np.array) -> int:
        """Check the following for four identical pieces:
        - Row in X (16)
        - Row in Y (16)
        - Column (16)
        - Flat diagonals (8)
        - Row bottom to top diagonals in X (8)
        - Row bottom to top diagonals in Y (8)
        - 3D Diagonals (4)

        Total: 76
        """
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

        for layer in board:
            for i in range(4):
                if self._check_identical(layer[i, :]):  # Every row in X (16)
                    return layer[i, 0]
                if self._check_identical(layer[:, i]):  # Every row in Y (16)
                    return layer[0, i]
            if self._check_identical(layer.ravel()[flat_diag1]):
                return layer.ravel()[flat_diag1[0]]  # Flat diagonals (8) [1/2]
            if self._check_identical(layer.ravel()[flat_diag2]):
                return layer.ravel()[flat_diag1[0]]  # Flat diagonals (8) [2/2]
        for i in range(4):
            if self._check_identical(
                board.ravel()[diag_x1 + diag_x_offset * i]
            ):  # Row bottom to top diagonals in X (8) [1/2]
                return board.ravel()[diag_x1 + diag_x_offset * i][0]
            if self._check_identical(
                board.ravel()[diag_x2 + diag_x_offset * i]
            ):  # Row bottom to top diagonals in X (8) [2/2]
                return board.ravel()[diag_x2 + diag_x_offset * i][0]

            if self._check_identical(
                board.ravel()[diag_y1 + diag_y_offset * i]
            ):  # Row bottom to top diagonals in Y (8) [1/2]
                return board.ravel()[diag_y1 + diag_y_offset * i][0]
            if self._check_identical(
                board.ravel()[diag_y2 + diag_y_offset * i]
            ):  # Row bottom to top diagonals in Y (8) [2/2]
                return board.ravel()[diag_y2 + diag_y_offset * i][0]
        for diag in diag_3d:
            if self._check_identical(board.ravel()[diag]):  # 3D Diagonals (4)
                return board.ravel()[diag][0]

    # def play(self, players: List[Player]):
    #     self.players = players
    #     self._create_board()

    #     # player = np.random.choice(players, 1)[0] # Start turn. Maybe always player 1
    #     player = players[0]

    #     while True:

    def _debug(self):
        self.board[0] = np.array(
            [[0, 1, -1, 1], [-1, 1, 0, -1], [1, 1, 1, -1], [0, -1, 0, 0]]
        )

        self.board[1] = 2
        self.board[2] = 3
        self.board[3] = 4
        # self.board[3][2][1] = 0


CF = ConnectFour3D()
CF._debug()
# print(CF.board)

# for layer in CF.board:
#     layer.ravel()[11] = 32
#     # print(layer)
#     # print(layer.ravel()[11])
#     # print("")
print(CF.board)
# print(CF._get_available_slots())
print(CF._get_available_slots(CF.board))
# bd = CF.make_move(CF.board, piece_placement=0)
# print("*" * 50)

# print(CF.board)
# print("-" * 50)
# print(bd)
#%%
