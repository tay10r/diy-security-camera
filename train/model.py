import torch
import torch.nn as nn
from typing import Union

class Block(nn.Module):
    def __init__(self, inputs: int, outputs: int, kernel_size: Union[int, tuple]):
        super().__init__()
        self.__layers = nn.Sequential(
            nn.Conv2d(inputs, outputs, kernel_size=kernel_size),
            nn.BatchNorm2d(outputs),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.__layers(x)

class SparseBlock(nn.Module):
    def __init__(self, in_filters: int, num_filters: int, num_filters_2: int, kernel_size: int):
        super().__init__()
        self.__layers = nn.Sequential(
            nn.Conv2d(in_filters, num_filters, kernel_size=kernel_size),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(num_filters, num_filters_2, kernel_size=1)
        )

    def forward(self, x):
        return self.__layers(x)

class PersonDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.__conv_layers = nn.Sequential(
            Block(1, 16, kernel_size=5),  # output_shape=(118, 158, K)
            Block(16, 24, kernel_size=3), # output_shape=(58, 78, K)
            Block(24, 32, kernel_size=3), # output_shape=(28, 38, K)
            Block(32, 48, kernel_size=(5, 7)), # output_shape=(12, 16, K)
            Block(48, 64, kernel_size=(5, 7)), # output_shape=(4, 5, K)
            nn.Conv2d(64, 1, kernel_size=1),
            #nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.__linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 5, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.__conv_layers(x)
        x = self.__linear_layers(x)
        return x
