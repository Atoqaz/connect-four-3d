from connectFour3D import ConnectFour3D
from tictactoe import TicTacToe
from player import Player
from player_functions import *

from profiler import profile


@profile
def measure_time(players, n_plays):
    connect_four = ConnectFour3D()
    for n in range(n_plays):
        connect_four.play(players, display=False)


def make_statistics(players, n_plays):
    wins = {}
    connect_four = ConnectFour3D()
    for n in range(n_plays):
        winner = connect_four.play(players, display=False)
        if winner is None:
            func_name = "Tied"
        else:
            func_name = winner.function.__name__
        if func_name not in wins:
            wins[func_name] = 1
        else:
            wins[func_name] += 1

    wins = {
        k: v for k, v in sorted(wins.items(), key=lambda item: item[1], reverse=True)
    }

    print(wins)


if __name__ == "__main__":
    players = [
        Player(name="A", function=rl_model_highest, piece_value=1),
        Player(name="B", function=None, piece_value=2),
    ]

    tic_tac_toe = TicTacToe()
    tic_tac_toe.play(players=players, display=True)

    # connect_four = ConnectFour3D()
    # connect_four.play(players=players, display=True)


    # make_statistics(players, n_plays=100_000)
    # measure_time(players, N=162_000)

    # print(players[0].function.__name__)
