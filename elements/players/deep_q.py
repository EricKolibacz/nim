"""Implements the deep q algorithm into a player"""
import copy
import random
from typing import List

import numpy as np
from torch import nn, no_grad, optim
from torch.utils.data import DataLoader

from elements.nim.action import Action
from elements.nim.board import Board
from elements.players.neural_network import DEVICE, BoardDataSet, NeuralNetwork, train
from elements.players.player_interfaces import AIQ, Player


class AI_V3(AIQ):
    """This AI does not update the Q-table after each step but once for all action_states
    after the entire game has finished. This increases the Learning curve a little bit."""

    def __init__(self, name, board: Board, alpha, gamma, epsilon) -> None:
        super().__init__(name, alpha, gamma, epsilon)
        self.max_number_of_stones = max(board.position)
        self.main_network: NeuralNetwork = NeuralNetwork(board).to(DEVICE)
        self.target_network = copy.deepcopy(self.main_network)

        self._loss_fn = nn.HuberLoss(delta=1.0)
        self._optimizer = optim.SGD(self.main_network.parameters(), lr=alpha)
        self.replay_buffer: List[SARS] = []

        self.n_star = 1

        self.steps = 0
        self.batch_size = 2**2
        self.buffer_length = 2**11
        self.update_after_steps = 2**7

    def _get_possible_actions(self, board: Board) -> Action:
        with no_grad():
            q_values = self.main_network(board.position)
            actions: list[Action] = [
                self._get_action_from_index(i, action) for i, action in self._extract_legal_moves(board, q_values)
            ]

        return actions

    def _extract_legal_moves(self, board: Board, q_values) -> List[int]:
        valid_q_values = []
        for i, q_value in enumerate(q_values):
            number_of_stones = i % self.max_number_of_stones + 1
            pile = int(i / self.max_number_of_stones)
            if board.position[pile] - number_of_stones >= 0:
                valid_q_values.append([i, q_value])
        return valid_q_values

    def _get_action_from_index(self, i: int, q_value: float) -> List[int]:
        number_of_stones = i % self.max_number_of_stones + 1
        pile = int(i / self.max_number_of_stones)
        return Action(number_of_stones, pile, q_value)

    def _update_q(self, board: Board, action: Action):
        if self.replay_buffer:
            if self.replay_buffer[-1].reward == 0:
                self.replay_buffer[-1].new_board = copy.deepcopy(board)
                # if self.replay_buffer.count(self.replay_buffer[-1]) > 1:
                #    self.replay_buffer.pop()

        self.replay_buffer.append(SARS(copy.deepcopy(board), copy.deepcopy(action), 0))

    def evaluate_result(self, has_won: bool):
        Player.evaluate_result(self, has_won)
        reward = 1 if has_won else -1
        if self.replay_buffer:
            self.replay_buffer[-1].reward = reward
            # if self.replay_buffer.count(self.replay_buffer[-1]) > 1:
            #    self.replay_buffer.pop()
            self.replay_buffer = self.replay_buffer[-self.buffer_length :]

            if self.batch_size < len(self.replay_buffer):
                input_batch = random.choices(self.replay_buffer, k=self.batch_size)
                x, y, action_ids = [], [], []
                for sars in input_batch:
                    if sars.new_board is not None:
                        self.target_network.eval()
                        actions = self.target_network(sars.new_board.position).detach().numpy()
                        max_q = max([q_value for _, q_value in self._extract_legal_moves(sars.new_board, actions)])
                    else:
                        max_q = 0
                    x.append(sars.board)
                    y.append(sars.reward + self.gamma * max_q)
                    action_ids.append(sars.action.pile * self.max_number_of_stones + sars.action.no_of_stones - 1)

                data_loader = DataLoader(BoardDataSet(x, y, action_ids), self.batch_size)
                self.main_network.train()
                train(data_loader, self.main_network, self._loss_fn, self._optimizer)
                self.main_network.eval()
            if (self.steps + 1) % self.update_after_steps == 0:
                self.target_network = copy.deepcopy(self.main_network)

            self.epsilon = 1 / np.sqrt(self.steps / +1)

            self.steps += 1
        # print(self.replay_buffer)
        # for item in self.replay_buffer:
        #    print(item)
        # input()


class SARS:
    """Class which implements the state-action-reward-state idea"""

    def __init__(self, board, action, reward: float, new_board=None) -> None:
        self.board: Board = board
        self.action: Action = action
        self.reward: float = reward
        self.new_board: Board = new_board

    def __eq__(self, __o: object) -> bool:
        return (
            self.board == __o.board
            and self.action == __o.action
            and self.reward == __o.reward
            and self.new_board == __o.new_board
        )

    def __repr__(self) -> str:
        return f"({self.board}, {self.action}, {self.reward}, {self.new_board})"
