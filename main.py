from copy import deepcopy

from elements.board import Board
from elements.game import play_game
from elements.players.ai_q import AI, AI_V2
from elements.players.deep_q import AI_V3
from elements.players.human import Human
from elements.players.random import Random

RUNS = 20000
PRINT_EVERY = 500

board = Board([1, 3, 5, 7])

random_player = Random("Rudolph")
# human player defined below
ai_q_player = AI("Dolly", 0.5, 0.95, 0.05)
ai_q_player2 = AI_V2("Dolly2", 0.5, 0.95, 0.05)
deep_q_player = AI_V3("Alex", board, 2e-4, 0.95, 0.85)

player2 = ai_q_player2
player3 = random_player


for i in range(RUNS):
    player2, player3 = play_game([player2, player3], deepcopy(board), 0)
    if (i + 1) % PRINT_EVERY == 0:
        print(
            f"{i+1:05}: {player2.name}: won {player2.victories} - lost {player2.defeats} - started {player2.started} - ratio {player2.victories/(player2.victories+player2.defeats):.2}"
        )
        print(
            f"{i+1:05}: {player3.name}: won {player3.victories} - lost {player3.defeats} - started {player3.started} - ratio {player3.victories/(player3.victories+player3.defeats):.2}"
        )
        player2.victories, player2.defeats, player2.started = 0, 0, 0
        player3.victories, player3.defeats, player3.started = 0, 0, 0


is_playing = "y" == input(f"Do you want to play against {player2.name}? (y/N) ")
if is_playing:
    name = input("What is your name? ")
    player1 = Human(name)
    while is_playing:
        player1, player2 = play_game([player1, player2], deepcopy(board), verbosity=2)
        is_playing = "y" == input("Do you want to continue playing? (y/N) ")
