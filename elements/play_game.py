"""Module containing the function to play the game"""
import copy
import random
from typing import List

from elements.board import Board
from elements.nim import Nim
from elements.players.player_interfaces import Player


def play_game(players: Player, board: Board, verbosity: int = 0) -> List[Player]:
    """Function to play the game"""
    game = Nim(board)
    current_player = random.choice(players)
    current_player.is_starting()

    while not game.has_ended():
        if verbosity >= 2:
            print(f"It is {current_player.name}'s move.")
        try:
            if verbosity >= 2:
                print(f"Current board setup: {game.board}")
            action = current_player.choose_next_action(copy.deepcopy(game.board))
            if verbosity >= 2:
                print(f"Taking {action.no_of_stones} stones from pile {action.pile}")
            game.remove(action)
        except ValueError as error_message:
            print("Encountered Error: ", error_message)
        else:
            current_player = players[1] if current_player == players[0] else players[0]
            if verbosity >= 2:
                print()
    for player in players:
        player.evaluate_result(current_player == player)
    if verbosity >= 2:
        print(f"{current_player.name} won.")

    return players
