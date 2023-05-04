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

    def _show_board_places(self, available_slots: np.array) -> None:
        slots = np.reshape(range(16), (4, 4))
        return np.reshape(
            [x if x in available_slots else "-" for x in slots.ravel()], (4, 4)
        )

    def get_available_slots(self, board: np.array) -> np.array:
        """List from 0 to 15 (inclusive) of the slots where placement is available"""
        return [
            int(i)
            for i, b in zip(range(16), (board[-1] == 0).astype(int).ravel())
            if b == 1
        ]

    def make_move(
        self, board: np.array, player: Player, piece_placement: int
    ) -> np.array:
        available_slots = self.get_available_slots(board=self.board)
        if piece_placement not in available_slots:
            raise ValueError(
                f"The selected piece placement is not available. Chosen placement: {piece_placement}.\nAvailable: {available_slots}"
            )

        for layer in board:
            location = layer.ravel()[piece_placement]
            if location == 0:
                layer.ravel()[piece_placement] = player.piece_value
                return board

    def _check_identical(self, list: np.array) -> bool:
        return (list[0] != 0) & (list == list[0]).all()

    def _detect_win(self, board: np.array) -> bool:
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
                    return True
                if self._check_identical(layer[:, i]):  # Every row in Y (16)
                    return True
            if self._check_identical(layer.ravel()[flat_diag1]):
                return True  # Flat diagonals (8) [1/2]
            if self._check_identical(layer.ravel()[flat_diag2]):
                return True  # Flat diagonals (8) [2/2]
        for i in range(4):
            for j in range(4):
                if self._check_identical(board[:, i, j]):  # Column (16)
                    return True
            if self._check_identical(
                board.ravel()[diag_x1 + diag_x_offset * i]
            ):  # Row bottom to top diagonals in X (8) [1/2]
                return True
            if self._check_identical(
                board.ravel()[diag_x2 + diag_x_offset * i]
            ):  # Row bottom to top diagonals in X (8) [2/2]
                return True

            if self._check_identical(
                board.ravel()[diag_y1 + diag_y_offset * i]
            ):  # Row bottom to top diagonals in Y (8) [1/2]
                return True
            if self._check_identical(
                board.ravel()[diag_y2 + diag_y_offset * i]
            ):  # Row bottom to top diagonals in Y (8) [2/2]
                return True
        for diag in diag_3d:
            if self._check_identical(board.ravel()[diag]):  # 3D Diagonals (4)
                return True
        return False

    def play(self, players: List[Player], display: bool = False) -> int:
        self._create_board()

        turn = 0

        while True:  # Game is running
            player = players[turn]

            available_slots = self.get_available_slots(board=self.board)
            if available_slots:

                while True:  # Get correct input
                    if player.function == None:
                        try:
                            print(
                                f"Turn = Player: {player.name}\nPiece: {player.piece_value}"
                            )
                            place_in_slot = int(
                                input(
                                    f"Select slot for the piece:\n{self._show_board_places(available_slots)}: "
                                )
                            )
                        except ValueError:
                            continue
                    else:
                        place_in_slot = player.function(
                            board=self.board, available_slots=available_slots,
                        )

                    if place_in_slot in available_slots:
                        break
                    else:
                        print("*" * 50)
                        print("Incorrect input")
                        print("*" * 50)

                self.board = self.make_move(
                    board=self.board, player=player, piece_placement=place_in_slot
                )

                if display:
                    print(self.board)

                if self._detect_win(board=self.board):
                    if display:
                        print(f"Game won by player: {player.name}")
                    return player

                turn = (turn + 1) % len(players)  # Next player in turn
            else:
                if display:
                    print("Game tied!")
                return None


# CF = ConnectFour3D()
