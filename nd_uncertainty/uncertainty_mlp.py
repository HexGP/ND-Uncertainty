"""
Uncertainty MLP

Small MLP that maps DINO patch embeddings to per-ray uncertainty β(r).
Mirrors NeRF-on-the-Go's UncerMLP architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyMLP(nn.Module):
    """
    MLP G(r) that maps DINO patch embeddings to per-ray uncertainty β(r).

    Mirror NeRF-on-the-Go:
      - Input dim = patch embedding dim (C * patch_size^2)
      - 1 hidden layer with ReLU activation
      - Dropout for regularization
      - Output dim = 1 (scalar β per ray)
      - Ensure β(r) > 0 (softplus + epsilon)

    NeRF-on-the-Go uses:
      - hidden_dim = 62
      - dropout_rate = 0.25
      - softplus activation on output
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        dropout_rate: float = 0.25,
    ):
        """
        Args:
            in_dim: Input dimension (patch embedding size = C * patch_size^2).
            hidden_dim: Hidden layer dimension. NeRF-on-the-Go uses 62, but
                       we default to 64 for flexibility.
            dropout_rate: Dropout rate for regularization (default 0.25).
        """
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # Single hidden layer MLP
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Initialize weights (He uniform, similar to NeRF-on-the-Go)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, patches: torch.Tensor, is_training: bool = None) -> torch.Tensor:
        """
        Forward pass: patches → β(r).

        Args:
            patches: (B, R, C_patch) DINO patch embeddings.
            is_training: Whether in training mode (for dropout).
                        If None, uses self.training.

        Returns:
            beta: (B, R, 1) positive uncertainty values β(r).
        """
        if is_training is None:
            is_training = self.training

        # Hidden layer with ReLU
        x = self.fc1(patches)  # (B, R, hidden_dim)
        x = F.relu(x)
        x = self.dropout(x) if is_training else x

        # Output layer
        x = self.fc2(x)  # (B, R, 1)

        # Apply softplus to ensure positive output, add small epsilon for stability
        beta = F.softplus(x) + 1e-6

        return beta
