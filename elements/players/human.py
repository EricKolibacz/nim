"""Defines the human player"""
from argparse import Action

from elements.board import Board
from elements.players.player_interfaces import Player


class Human(Player):
    def choose_next_action(self, board: Board) -> Action:
        is_inputing = True
        while is_inputing:
            pile = int(input("From which pile do you want to remove stone(s)?  "))
            no_of_stones = int(input("How many stones?  "))
            try:
                action = Action(no_of_stones, pile)
            except ValueError as excpetion:
                print(excpetion)

        return action
