"""
Uncertainty MLP

Small MLP that maps DINO patch embeddings to per-ray log-variance s(r) = log σ(r).
Per ND-SDF principles: predict log-variance for numerical stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyMLP(nn.Module):
    """
    MLP G(r) that maps DINO patch embeddings to per-ray log-variance s(r) = log σ(r).

    Per ND-SDF implementation notes:
      - Predict log-variance s(r) = log σ_c(r) for numerical stability
      - Variance is computed as σ_c^2 = exp(2s)
      - σ is clamped to [σ_min, σ_max] to prevent extreme down-weighting
      - Initialize log σ to s_0 = -3 (σ ≈ 0.05) to prevent underfitting

    Architecture:
      - Input dim = patch embedding dim (C * patch_size^2)
      - 1 hidden layer with ReLU activation
      - Dropout for regularization
      - Output dim = 1 (scalar s = log σ per ray)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        dropout_rate: float = 0.25,
        init_log_sigma: float = -3.0,  # s_0 = -3 → σ ≈ 0.05
        sigma_min: float = 1e-3,
        sigma_max: float = 0.5,
    ):
        """
        Args:
            in_dim: Input dimension (patch embedding size = C * patch_size^2).
            hidden_dim: Hidden layer dimension (default 64).
            dropout_rate: Dropout rate for regularization (default 0.25).
            init_log_sigma: Initial value for log σ (s_0, default -3.0 → σ ≈ 0.05).
            sigma_min: Minimum σ value for clamping (default 1e-3).
            sigma_max: Maximum σ value for clamping (default 0.5).
        """
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.init_log_sigma = init_log_sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Single hidden layer MLP
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Initialize weights (He uniform)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        
        # Initialize output bias to s_0 = -3 (log σ ≈ -3 → σ ≈ 0.05)
        # This prevents underfitting from large initial σ values
        nn.init.constant_(self.fc2.bias, init_log_sigma)

    def forward(self, patches: torch.Tensor, is_training: bool = None) -> torch.Tensor:
        """
        Forward pass: patches → s(r) = log σ(r) → σ(r).

        Per ND-SDF implementation:
          - Predict log-variance s(r) = log σ_c(r)
          - Compute σ = exp(s), then clamp to [σ_min, σ_max]
          - This ensures numerical stability and prevents extreme down-weighting

        Args:
            patches: (B, R, C_patch) DINO patch embeddings.
            is_training: Whether in training mode (for dropout).
                        If None, uses self.training.

        Returns:
            sigma: (B, R, 1) clamped uncertainty values σ(r) ∈ [σ_min, σ_max].
        """
        if is_training is None:
            is_training = self.training

        # Hidden layer with ReLU
        x = self.fc1(patches)  # (B, R, hidden_dim)
        x = F.relu(x)
        x = self.dropout(x) if is_training else x

        # Output layer: predicts log-variance s(r) = log σ(r)
        log_sigma = self.fc2(x)  # (B, R, 1)

        # Compute σ = exp(s) for numerical stability
        sigma = torch.exp(log_sigma)  # (B, R, 1)

        # Clamp σ to [σ_min, σ_max] to prevent extreme down-weighting
        # This prevents the loss from being down-weighted too much when σ is very large
        sigma = sigma.clamp(min=self.sigma_min, max=self.sigma_max)  # (B, R, 1)
        
        # Debug: Print statistics on first forward pass (only once)
        if not hasattr(self, '_debug_printed'):
            with torch.no_grad():
                print(f"[UncertaintyMLP Debug] First forward pass:")
                print(f"  - log_sigma (s): min={log_sigma.min().item():.4f}, max={log_sigma.max().item():.4f}, mean={log_sigma.mean().item():.4f}")
                print(f"  - sigma (before clamp): min={torch.exp(log_sigma).min().item():.4f}, max={torch.exp(log_sigma).max().item():.4f}")
                print(f"  - sigma (after clamp): min={sigma.min().item():.4f}, max={sigma.max().item():.4f}, mean={sigma.mean().item():.4f}")
                print(f"  - fc2.bias (s_0): {self.fc2.bias.item():.4f}")
            self._debug_printed = True

        return sigma
