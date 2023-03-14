"""This module contains the interfaces for the players."""
import random
from abc import ABC, abstractmethod
from typing import List

from elements.action import Action
from elements.board import Board


class Player(ABC):
    def __init__(self, name) -> None:
        self.name = name
        self.victories = 0
        self.defeats = 0
        self.started = 0

    @abstractmethod
    def choose_next_action(self, board: List) -> Action:
        """choose the next action given the board

        Args:
            board (List): the current board

        Returns:
            List: number of stones, pile number
        """

    def evaluate_result(self, has_won: bool):
        if has_won:
            self.victories += 1
        else:
            self.defeats += 1

    def is_starting(self):
        self.started += 1

    def __eq__(self, __o: object) -> bool:
        return self.name == __o.name


class AIQ(Player, ABC):
    def __init__(self, name, alpha, gamma, epsilon) -> None:
        super().__init__(name)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    @abstractmethod
    def _get_possible_actions(self, board: Board) -> dict:
        pass

    @abstractmethod
    def _update_q(self, board: Board, action: Action):
        pass

    def choose_next_action(self, board: Board) -> Action:
        actions: List[Action] = self._get_possible_actions(board)
        if random.uniform(0.0, 1.0) >= self.epsilon:
            next_action = max(actions, key=lambda act: act.q_value)
        else:
            next_action = random.choice(actions)

        self._update_q(board, next_action)

        return next_action
