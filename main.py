from copy import deepcopy

from elements.board import Board
from elements.game import play_game
from elements.players.ai_q import AI
from elements.players.deep_q import AI_V3
from elements.players.random import Random

RUNS = 100000000
PRINT_EVERY = 500

board = Board([1, 3, 5, 7])

player2 = AI("Dolly", 0.5, 0.95, 0.05)  # Random("Rudolf")  #
player1 = AI_V3("Alex", board, 2e-4, 0.95, 0.85)  # , is_initializing_q_randomly=False)
# player1 = AI("Dolly", 0.5, 0.95, 0)
# player2 = AI("Alex", 0.5, 0.95, 0)  # , is_initializing_q_randomly=False)


for i in range(RUNS):
    player1, player2 = play_game([player1, player2], deepcopy(board), 0)  # 2 if i > 500 else 0)
    if (i + 1) % PRINT_EVERY == 0:
        print(
            f"{i+1:05}: {player1.name}: won {player1.victories} - lost {player1.defeats} - illegal {player1.lost_of_illegal_move} - started {player1.started} - ratio {player1.victories/(player1.victories+player1.defeats):.2}"
        )
        print(
            f"{i+1:05}: {player2.name}: won {player2.victories} - lost {player2.defeats} - illegal {player2.lost_of_illegal_move} - started {player2.started} - ratio {player2.victories/(player2.victories+player2.defeats):.2}"
        )
        print(player1.epsilon)
        tmp_board = Board([0, 3, 2, 0])
        # actions1 = player1._get_next_actions(tmp_board)
        actions1 = player1._get_possible_actions(tmp_board)
        # actions2 = {
        #    action: q for action, q in actions2.items() if tmp_board[action[1]] - action[0] >= 0
        # }
        for action in actions1:
            print(f"{action}")  # -> {actions1[key]:.2f}")

        if player1.victories / (player1.victories + player1.defeats) > 1.4:
            break
        player1.victories = 0
        player1.defeats = 0
        player1.started = 0
        player1.lost_of_illegal_move = 0
        player2.victories = 0
        player2.defeats = 0
        player2.started = 0
        player2.lost_of_illegal_move = 0


is_playing = False  # "y" == input(f"Do you want to play against {player1.name}? (y/N) ")
if is_playing:
    name = input("What is your name?")
    player3 = Human(name)
    while is_playing:
        player3, player2 = play_game([player3, player1], deepcopy(board), verbosity=2)
        is_playing = "y" == input("Do you want to continue playing? (y/N) ")
