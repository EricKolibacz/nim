"""Define model"""

from typing import List

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from elements.board import Board

DEVICE = "cpu"


class NeuralNetwork(nn.Module):
    def __init__(self, board: Board):
        super().__init__()
        max_number_of_stones = max(board.position)
        input_shape = int(np.log2(max_number_of_stones) + 1) * board.no_of_piles
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, max_number_of_stones * board.no_of_piles),
            # nn.Sigmoid(),
        )
        self.mask = 2 ** torch.arange(input_shape // board.no_of_piles)

    def forward(self, x):
        x = torch.as_tensor(x).unsqueeze(-1).bitwise_and(self.mask).ne(0).byte()
        x = x.flatten(start_dim=len(x.size()) - 2).float().to(DEVICE)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for X, y, action_id in dataloader:
        y = y.to(DEVICE)

        # Compute prediction error
        pred = model(X)
        specific_pred = pred[np.arange(len(pred)), action_id.long()]
        loss = loss_fn(specific_pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class BoardDataSet(Dataset):
    """A costume torch-DataSet class"""

    def __init__(self, boards: List[Board], q_values: List[float], action_ids: List[np.ndarray]) -> None:
        self.boards = boards
        self.q_values = Tensor(q_values)
        self.action_ids = Tensor(action_ids)

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, index: int):
        return torch.as_tensor(self.boards[index].position), self.q_values[index], self.action_ids[index]
