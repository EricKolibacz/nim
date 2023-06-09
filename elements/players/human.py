"""Defines the human player"""

from elements.nim.action import Action
from elements.nim.board import Board
from elements.players.player_interfaces import Player


class Human(Player):
    def choose_next_action(self, board: Board) -> Action:
        is_inputing = True
        while is_inputing:
            pile = int(input("From which pile do you want to remove stone(s)?  "))
            no_of_stones = int(input("How many stones?  "))
            try:
                action = Action(no_of_stones, pile)
                is_inputing = False
            except ValueError as excpetion:
                print(excpetion)

        return action
