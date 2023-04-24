from connectFour3D import ConnectFour3D, Player
from player_functions import *


if __name__ == "__main__":
    players = [
        Player(name="A", function=None, piece_value=1),
        Player(name="B", function=None, piece_value=2),
    ]

    CF = ConnectFour3D()
    CF.play(players=players, display=True)
