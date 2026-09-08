"""Tiny CNN black box for the digit-image integration tests.

The architecture lives in its own module so ``generate_artifacts.py`` (which
trains the models once) and the test suite (which loads the committed
state_dicts) share a single definition.
"""

from torch import nn


class TinyCNN(nn.Module):
    def __init__(self, n_classes=10, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, n_classes),
        )

    def forward(self, x):
        return self.net(x)
