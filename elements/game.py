import copy
import random
from typing import List

from elements.board import Board
from elements.players.player_interfaces import Player
from elements.players.random import Random
from nim import Nim


def play_game(players: Player, board: Board, verbosity: int = 0) -> List[Player]:
    game = Nim(board)
    current_player = random.choice(players)
    current_player.is_starting()

    has_lost_of_illegal_move: Player = Random("ReallyRandomy")

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
            if "NIM_ERROR:" in error_message.args[0]:
                current_player = players[1] if current_player == players[0] else players[0]
                not_current_player = players[1] if current_player == players[0] else players[0]
                has_lost_of_illegal_move = not_current_player
                break
            else:
                print("Encountered Error: ", error_message)
            input()
        else:
            current_player = players[1] if current_player == players[0] else players[0]
            if verbosity >= 2:
                print()
    for player in players:
        player.evaluate_result(current_player == player, has_lost_of_illegal_move == player)
    if verbosity >= 2:
        print(f"{current_player.name} won.")

    return players
