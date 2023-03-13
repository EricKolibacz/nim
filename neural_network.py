"""Define model"""

from typing import List

import numpy as np
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cpu"


class NeuralNetwork(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_shape),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)  # * 2 - 1
        return logits


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for X, y, action_id in dataloader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Compute prediction error
        pred = model(X)
        specific_pred = pred[np.arange(len(pred)), action_id.long()]
        loss = loss_fn(specific_pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def create_dataloader(
    x: np.ndarray, y: np.ndarray, action_ids: np.ndarray, batch_size: int
) -> DataLoader:
    dataset = TensorDataset(Tensor(x), Tensor(y), Tensor(action_ids))  # create dataset
    return DataLoader(dataset, batch_size)
