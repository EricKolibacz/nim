"""the Nim game"""


from elements.nim.action import Action
from elements.nim.board import Board


class Nim:
    """Nim game class"""

    def __init__(self, board: Board) -> None:
        self.board = board

    def remove(self, action: Action) -> None:
        """Removes the number of stones from the pile defined in action.
        In other words, the method executes the action."""
        if action.pile > self.board.no_of_piles + 1 or action.pile < 0:
            raise ValueError("NIM_ERROR: Index error for board pile. Needs to be between 0 and 3.")
        if self.board.position[action.pile] - action.no_of_stones < 0:
            raise ValueError("NIM_ERROR: You try to remove too many stones.")

        self.board.position[action.pile] -= action.no_of_stones
        self.board = Board(self.board.position)
        if self.has_ended():
            pass  # print("You lost the Game")

    def has_ended(self):
        """Determining if nim has ended."""
        return sum(self.board.position) == 0
