import numpy as np

"""

Create Board (done)
Get available slots 

Make move (isPlayerOneTurn: bool)

Create winning function

Display board

"""


class ConnectFour3D:
    def __init__(self):
        self.board = self._create_board()
        self.player_turn = 1

    def _create_board(self) -> np.array:
        return np.zeros([2, 4, 4], dtype=int)

    def _get_available_slots(self, board: np.array) -> np.array:
        return (board[-1] == 0).astype(int)

    def make_move(self, board: np.array, piece_placement: int, player_turn: int = None):
        if player_turn is None:
            player_turn = self.player_turn
        available_slots = self._get_available_slots(board=self.board).ravel()
        if not available_slots[piece_placement]:
            return board
        for layer in board:
            location = layer.ravel()[piece_placement]
            if location == 0:
                location = player_turn
                return board

    def _debug(self):
        self.board[0] = np.array(
            [[0, 1, -1, 1], [-1, 1, 0, -1], [1, 1, 1, -1], [0, -1, 0, 0]]
        )
        self.board[1] = 0
        # self.board[2] = 3
        # self.board[3] = 4
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
bd = CF.make_move(CF.board, piece_placement=0)
print("*" * 50)

print(CF.board)
print("-" * 50)
print(bd)
