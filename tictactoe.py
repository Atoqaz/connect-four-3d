"""

Create Board (done)
Get available slots 

Make move (isPlayerOneTurn: bool)

Create winning function

Display board

"""

# %%
import numpy as np
from typing import List
import os

from player import Player

class TicTacToe:
    """Players: 2 players with values 1 & 2. 0 is empty slot."""

    def __init__(self):
        self.width = 3
        self.pieces = self.width**2
        self.board = self._create_board()

    def _create_board(self) -> np.array:
        return np.zeros([self.width, self.width], dtype=int)

    def _show_board_places(self, available_slots: np.array) -> None:
        # slots = np.reshape(range(self.pieces), (self.width, self.width))
        return np.reshape(
            [x if x in available_slots else "-" for x in range(self.pieces)],
            (self.width, self.width),
        )

    def get_available_slots(self, board: np.array) -> np.array:
        """List from 0 to number of pieces (inclusive) of the slots where placement is available"""
        return [
            int(i)
            for i, b in zip(range(self.pieces), (board == 0).astype(int).ravel())
            if b == 1
        ]

    def make_move(
        self, board: np.array, player: Player, piece_placement: int
    ) -> np.array:
        available_slots = self.get_available_slots(board=board)
        if piece_placement not in available_slots:
            raise ValueError(
                f"The selected piece placement is not available. Chosen placement: {piece_placement}.\nAvailable: {available_slots}"
            )

        location = board.ravel()[piece_placement]
        if location == 0:
            board.ravel()[piece_placement] = player.piece_value
            return board

    def _check_identical(self, list: np.array) -> bool:
        return (list[0] != 0) & (list == list[0]).all()

    def _detect_win(self, board: np.array) -> bool:
        """Check the following for four identical pieces:
        - Row in X (3)
        - Row in Y (3)
        - Diagonals (2)
        Total: 8
        """

        # board index
        diag_1 = np.array([0, 4, 8])  # diagonal top left to bottom right
        diag_2 = np.array([2, 4, 6])  # diagonal top right to bottom left

        for i in range(self.width):
            if self._check_identical(board[i, :]):  # Rows
                return True
            if self._check_identical(board[:, i]):  # Columns
                return True
        if self._check_identical(board.ravel()[diag_1]):
            return True
        if self._check_identical(board.ravel()[diag_2]):
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
                            piece_placement = int(
                                input(
                                    f"Select slot for the piece:\n{self._show_board_places(available_slots)}: "
                                )
                            )
                        except ValueError:
                            continue
                    else:
                        piece_placement = player.function(
                            board=self.board,
                            available_slots=available_slots,
                        )

                    if piece_placement in available_slots:
                        break
                    else:
                        print("*" * 50)
                        print("Incorrect input")
                        print("*" * 50)

                self.board = self.make_move(
                    board=self.board, player=player, piece_placement=piece_placement
                )

                if display:
                    os.system("cls")
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
