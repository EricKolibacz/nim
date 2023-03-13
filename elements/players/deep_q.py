"""Implements the deep q algorithm into a player"""
import copy
import random
from typing import List

import numpy as np
import torch
from torch import nn, no_grad, optim

from elements.action import Action
from elements.board import Board
from elements.players.player_interfaces import AIQ, Player
from neural_network import DEVICE, NeuralNetwork, create_dataloader, train


class AI_V3(AIQ):
    """This AI does not update the Q-table after each step but once for all action_states
    after the entire game has finished. This increases the Learning curve a little bit."""

    def __init__(self, name, board: Board, alpha, gamma, epsilon) -> None:
        super().__init__(name, alpha, gamma, epsilon)
        self.max_number_of_stones = max(board.position)
        self.input_size = int(np.log2(self.max_number_of_stones) + 1) * board.no_of_piles
        self.main_network: NeuralNetwork = NeuralNetwork(
            self.input_size,
            self.max_number_of_stones * board.no_of_piles,
        ).to(DEVICE)
        self.target_network = copy.deepcopy(self.main_network)

        self._loss_fn = nn.HuberLoss(delta=1.0)
        self._optimizer = optim.Adam(self.main_network.parameters(), lr=alpha)
        self.replay_buffer: List[SARS] = []

        self.n_star = 1

        self.steps = 0
        self.batch_size = 2**2
        self.buffer_length = 2**11
        self.update_after_steps = 2**7

    def _get_possible_actions(self, board: Board) -> Action:
        with no_grad():
            actions = self.main_network(torch.tensor(board.binary_reprensation, dtype=torch.float))
        return [self._get_action_from_index(i, action) for i, action in enumerate(actions)]

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

        self.replay_buffer[-1].reward = reward
        if self.has_lost_of_illegal_move:
            self.replay_buffer[-1].new_board = None
        # if self.replay_buffer.count(self.replay_buffer[-1]) > 1:
        #    self.replay_buffer.pop()
        self.replay_buffer = self.replay_buffer[-self.buffer_length :]

        if self.batch_size < len(self.replay_buffer):
            input_batch = random.choices(self.replay_buffer, k=self.batch_size)
            x, y, action_ids = [], [], []
            for sars in input_batch:
                if sars.new_board is not None:
                    self.target_network.eval()
                    actions = (
                        self.target_network(torch.tensor(sars.new_board.binary_reprensation, dtype=torch.float))
                        .detach()
                        .numpy()
                    )
                    max_q = max(actions)
                else:
                    max_q = 0
                x.append(torch.tensor(sars.board.binary_reprensation, dtype=torch.float).detach().numpy())
                y.append(sars.reward + self.gamma * max_q)
                action_ids.append(sars.action.pile * self.max_number_of_stones + sars.action.no_of_stones - 1)

            data_loader = create_dataloader(np.array(x), np.array(y), np.array(action_ids), self.batch_size)
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
