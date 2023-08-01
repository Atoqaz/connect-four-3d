
from dataclasses import dataclass

@dataclass()
class Player:
    name: str  # Unique
    function: callable
    piece_value: int = None

    def __repr__(self):
        return f"Player(Name = {self.name})"
