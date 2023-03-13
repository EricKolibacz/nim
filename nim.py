"""the Nim game"""


from elements.action import Action
from elements.board import Board


class Nim:
    def __init__(self, board: Board) -> None:
        self.board = board

    def remove(self, action: Action) -> None:
        if action.pile > self.board.no_of_piles + 1 or action.pile < 0:
            raise ValueError("NIM_ERROR: Index error for board pile. Needs to be between 0 and 3.")
        if self.board.position[action.pile] - action.no_of_stones < 0:
            raise ValueError("NIM_ERROR: You try to remove too many stones.")

        self.board.position[action.pile] -= action.no_of_stones
        self.board = Board(self.board.position)
        if self.has_ended():
            pass  # print("You lost the Game")

    def has_ended(self):
        return sum(self.board.position) == 0
