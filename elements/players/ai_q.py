"""Implements simple q learning"""
import copy
import random
from typing import List

from elements.action import Action
from elements.board import Board
from elements.players.player_interfaces import AIQ, Player


class AI(AIQ):
    def __init__(self, name, alpha, gamma, epsilon, is_initializing_q_randomly=True) -> None:
        super().__init__(name, alpha, gamma, epsilon)
        self.q_table = {}
        self._previous_action_state = None

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.is_initializing_q_randomly = is_initializing_q_randomly

    def _get_possible_actions(self, board: Board) -> List[Action]:
        if board not in self.q_table:
            self.q_table[board] = [
                Action(
                    removable_number_of_stones,
                    pile,
                    random.uniform(0.0, 0.01) if self.is_initializing_q_randomly else 0,
                )
                for pile, total_stones in enumerate(board.position)
                if total_stones > 0
                for removable_number_of_stones in range(1, total_stones + 1)
            ]

        return self.q_table[board]

    def _update_q(self, board: Board, action: Action):
        new_board: list = list(board.position)
        new_board[action.pile] -= action.no_of_stones
        reward = -1 if sum(new_board) == 0 else 0
        if self._previous_action_state is not None:
            prev_board, prev_action = self._previous_action_state
            index = self.q_table[prev_board].index(prev_action)
            self.q_table[prev_board][index].q_value = prev_action.q_value + self.alpha * (
                (reward + self.gamma * action.q_value) - prev_action.q_value
            )
        self._previous_action_state = (copy.deepcopy(board), copy.deepcopy(action))

    def evaluate_result(self, has_won: bool):
        super().evaluate_result(has_won)
        reward = 1 if has_won else -1
        if self._previous_action_state is not None:
            prev_board, prev_action = self._previous_action_state
            index = self.q_table[prev_board].index(prev_action)
            self.q_table[prev_board][index].q_value = prev_action.q_value + self.alpha * (reward - prev_action.q_value)
            self._previous_action_state = None


class AI_V2(AI):
    """This AI does not update the Q-table after each step but once for all action_states
    after the entire game has finished. This increases the Learning curve a little bit."""

    def __init__(self, name, alpha, gamma, epsilon, is_initializing_q_randomly=True) -> None:
        super().__init__(name, alpha, gamma, epsilon, is_initializing_q_randomly)
        self._previous_action_state = []

    def _update_q(self, board: Board, action: Action):
        self._previous_action_state.append((copy.deepcopy(board), copy.deepcopy(action)))

    def evaluate_result(self, has_won: bool):
        Player.evaluate_result(self, has_won)
        reward = 1 if has_won else -1

        prev_board, prev_action = self._previous_action_state[-1]
        index = self.q_table[prev_board].index(prev_action)

        self.q_table[prev_board][index].q_value = prev_action.q_value + self.alpha * (reward - prev_action.q_value)

        reversed_action_states = list(reversed(self._previous_action_state))
        reward = 0

        for action_state, previous_action_state in zip(reversed_action_states, reversed_action_states[1:]):
            action_utility = max(self.q_table[action_state[0]], key=lambda action: action.q_value).q_value
            prev_board, prev_action = previous_action_state
            index = self.q_table[prev_board].index(prev_action)
            self.q_table[prev_board][index].q_value = prev_action.q_value + self.alpha * (
                (reward + self.gamma * action_utility) - prev_action.q_value
            )

        self._previous_action_state = []
