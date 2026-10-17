
"""
Mamba-based EEG encoder for binary motor-imagery classification.

Input:
    (batch_size, 3, n_samples)

Output:
    (batch_size, 2)
    Class 0 = left hand
    Class 1 = right hand
"""

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError as exc:
    raise ImportError(
        "The Mamba package is unavailable. Install a compatible "
        "mamba-ssm build in your Colab environment, then restart "
        "the runtime if necessary."
    ) from exc


class EEGMambaEncoder(nn.Module):
    """
    Projects multichannel EEG samples into an embedding sequence,
    processes the sequence with Mamba blocks, and predicts the class.
    """

    def __init__(
        self,
        n_channels=3,
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        n_layers=2,
        n_classes=2,
        dropout=0.2,
    ):
        super().__init__()

        if n_channels != 3:
            raise ValueError(
                "This configuration expects 3 EEG channels: C3, Cz, C4."
            )
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1.")

        self.n_channels = n_channels
        self.d_model = d_model
        self.n_classes = n_classes

        # Each time point contains the simultaneous C3, Cz and C4 values.
        self.input_projection = nn.Linear(n_channels, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            EEG batch shaped (B, 3, T).

        Returns
        -------
        torch.Tensor
            Unnormalized class logits shaped (B, 2).
            Pass these directly to nn.CrossEntropyLoss.
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape (B, 3, T), got {tuple(x.shape)}."
            )

        if x.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, "
                f"got {x.shape[1]}."
            )

        # (B, 3, T) -> (B, T, 3)
        x = x.transpose(1, 2)

        # (B, T, 3) -> (B, T, d_model)
        x = self.input_projection(x)
        x = self.input_norm(x)

        for mamba_layer, norm in zip(self.mamba_layers, self.norms):
            residual = x
            x = mamba_layer(x)
            x = norm(residual + self.dropout(x))

        # Aggregate temporal features into one vector per trial.
        x = x.mean(dim=1)

        # Return two class logits per trial.
        return self.classifier(x)


# Convenient alias for code that prefers the shorter model name.
EEGMamba = EEGMambaEncoder
