"""Contains the random player"""


import random
from typing import List

from elements.action import Action
from elements.board import Board
from elements.players.player_interfaces import Player


class Random(Player):
    def choose_next_action(self, board: Board) -> Action:
        possible_actions = self._get_possible_actions(board)
        return random.choice(possible_actions)

    def _get_possible_actions(self, board: Board) -> List[Action]:
        return [
            Action(removable_number_of_stones, pile)
            for pile, total_stones in enumerate(board.position)
            if total_stones > 0
            for removable_number_of_stones in range(1, total_stones + 1)
        ]
