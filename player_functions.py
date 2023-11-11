import numpy as np
from typing import List
from stable_baselines3 import PPO
from pathlib import Path
from tictactoe import TicTacToe
from typing import Tuple, List
from player import Player

DIR = Path(__file__).parent
models = []
for p in DIR.joinpath("train").glob("*.zip"):
    models.append(int(str(p).split("\\")[-1].split(".")[0]))
if models:
    HIGHEST_MODEL = PPO.load(f"./train/{max(models)}")
else:
    HIGHEST_MODEL = None


def place_random(board: np.array, available_slots: List[int], players: List[Player]):
    return np.random.choice(available_slots)


def place_minimum(board: np.array, available_slots: List[int], players: List[Player]):
    return min(available_slots)


def rl_model_highest(
    board: np.array, available_slots: List[int], players: List[Player]
):
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


def _reward_max_min_tic_tac_toe(board: np.array, player_value: int) -> int:
    width = 3
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
            if u == player_value:
                player_val = c
            elif u == -player_value:  # Opponent = - player (1, -1) values for them
                opponent_val = c
        player_sum += player_val
        opponent_sum += opponent_val
        return player_sum, opponent_sum

    player_sum = 0
    opponent_sum = 0

    for i in range(width):
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
    return player_sum - opponent_sum


def _get_best_move(
    board: np.array, depth: int, players: List[Player], turn: int
) -> Tuple[int, float, np.array]:
    """Recursively get the best move, by simulating outcomes

    Args:
        board (np.array): Board with pieces
        depth (int): Recursive depth
        players (List[Player]): Players playing
        turn (int): Index in "players" reprecenting player or opponent turn

    Returns:
        best_move (int): Move with the best score
        best (float): Score for the best move
        Scoreboard (np.array): Scores associated with each move
            column 0: piece placement
            column 1: score for that piece placement
    """
    available_slots = TicTacToe.get_available_slots(
        board=board, n_pieces=9
    )  # 9 pieces for tic tac toe

    player = players[turn]

    n_moves = len(available_slots)
    scoreboard = np.zeros((n_moves, 2))
    if n_moves == 0:
        best_move = []
        best = _reward_max_min_tic_tac_toe(
            board=board, player_value=player.piece_value
        )  # OBS: assumption for 1, -1 pieces
        return best_move, best, scoreboard

    for piece_index, piece_placement in enumerate(
        available_slots
    ):  # Try all available moves
        board2 = TicTacToe.make_move(
            board=np.copy(board),
            available_slots=available_slots,
            piece_placement=piece_placement,
            player=player,
        )
        # print(
        #     f"depth: {depth}, piece_placement: {piece_placement}, available_slots: {available_slots}, piece_index: {piece_index}, board: \n{board}\nboard2: \n{board2}"
        # )
        # if board is None:

        if depth >= 1:
            turn = (turn + 1) % len(players)
            _, best, _ = _get_best_move(
                board=board2, depth=depth - 1, players=players, turn=turn
            )
        else:
            best = _reward_max_min_tic_tac_toe(
                board=board2, player_value=player.piece_value
            )

        scoreboard[piece_index, :] = (piece_placement, best)

    best_move = int(scoreboard[np.argmax(scoreboard[:, 1]), 0])

    # _max = player*np.max(player*scoreboard[:,1]) # maximum of best score (player dependent)
    # score_diff = np.abs(scoreboard[:,1] - _max)
    # weight = 10**(-score_diff*2)
    # best=fix(scoreboard(ranchoice,2))+(user*(sum(scorediff<1)-1)/double(nmoves));
    # best_move=scoreboard(ranchoice,1);
    return best_move, best, scoreboard


def recursive(board: np.array, available_slots: List[int], players: List[Player]):
    best_move, _, _ = _get_best_move(board=board, depth=6, players=players, turn=0)
    return best_move
