"""
Model architecture for the Randles-CPE parameter regressor.

Two classes:
  - MLP: the trainable regressor. Input = 40-dim standardized [Re(Z) x20,
    Im(Z) x20] vector. Output = 4-dim transformed-scale targets
    [log10(Rs), log10(Rct), log10(Q), logit(alpha)].
  - ExportWrapper: wraps a trained MLP with baked-in input normalization
    (stored train-set mean/std, as buffers so they travel with the ONNX
    graph) and baked-in inverse transforms (10^x, sigmoid), so the exported
    graph takes raw Re/Im ohms in and returns raw-scale Rs/Rct/Q/alpha out.
"""

from __future__ import annotations

import torch
from torch import nn

N_INPUT = 40  # 20 Re(Z) + 20 Im(Z)
N_OUTPUT = 4  # log10(Rs), log10(Rct), log10(Q), logit(alpha)


class MLP(nn.Module):
    """40 -> 64 -> 64 -> 32 -> 4, ReLU activations, linear output head."""

    def __init__(self, hidden_sizes: tuple[int, ...] = (64, 64, 32)):
        super().__init__()
        sizes = (N_INPUT,) + tuple(hidden_sizes)
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[-1], N_OUTPUT))
        self.net = nn.Sequential(*layers)

    def forward(self, x_standardized: torch.Tensor) -> torch.Tensor:
        return self.net(x_standardized)


class ExportWrapper(nn.Module):
    """Normalization + MLP + inverse-transform, all as one ONNX graph.

    Output column order is fixed: [Rs, Rct, Q, alpha], raw scale (ohm, ohm,
    S*s^alpha, dimensionless in (0,1)).
    """

    def __init__(self, mlp: MLP, feature_mean: torch.Tensor, feature_std: torch.Tensor):
        super().__init__()
        self.mlp = mlp
        self.register_buffer("feature_mean", feature_mean.clone().float())
        self.register_buffer("feature_std", feature_std.clone().float())

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        x_standardized = (x_raw - self.feature_mean) / self.feature_std
        y_transformed = self.mlp(x_standardized)  # (..., 4): log_Rs, log_Rct, log_Q, logit_alpha
        log_Rs = y_transformed[..., 0]
        log_Rct = y_transformed[..., 1]
        log_Q = y_transformed[..., 2]
        logit_alpha = y_transformed[..., 3]

        Rs = 10.0**log_Rs
        Rct = 10.0**log_Rct
        Q = 10.0**log_Q
        alpha = torch.sigmoid(logit_alpha)

        return torch.stack([Rs, Rct, Q, alpha], dim=-1)
