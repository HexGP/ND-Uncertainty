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
        # Initialize output bias
        # NOTE: NeRF-on-the-Go uses default zero bias (Flax nn.Dense default).
        # However, with zero bias, if hidden layer outputs are small/zero (common with ReLU),
        # the final output can be very negative, making softplus ≈ 0 and beta ≈ 1e-6.
        # 
        # Options:
        # 1. bias=0 (matches NeRF-on-the-Go exactly, but may cause collapse)
        # 2. bias=1.5 (practical fix to ensure visible initial beta values)
        # 
        # Using bias=1.5 ensures softplus(1.5) ≈ 1.7, which is in visible range [0.2, 2.0].
        # This is NOT from the paper/repo, but a practical fix for the all-purple issue.
        # If you want to match NeRF-on-the-Go exactly, change this to: nn.init.zeros_(self.fc2.bias)
        nn.init.constant_(self.fc2.bias, 1.5)  # softplus(1.5) ≈ 1.7, visible as yellow/red

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

        # Apply softplus to ensure positive output
        # Note: We add a small epsilon AFTER softplus to ensure numerical stability,
        # but softplus already ensures positive values, so this is mainly for safety.
        beta = F.softplus(x) + 1e-6
        
        # Debug: Print beta statistics on first forward pass (only once)
        if not hasattr(self, '_debug_printed'):
            with torch.no_grad():
                print(f"[UncertaintyMLP Debug] First forward pass:")
                print(f"  - fc2 output (before softplus): min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}")
                print(f"  - beta (after softplus): min={beta.min().item():.4f}, max={beta.max().item():.4f}, mean={beta.mean().item():.4f}")
                print(f"  - fc2.bias value: {self.fc2.bias.item():.4f}")
            self._debug_printed = True

        return beta
