"""
models/architecture.py — CNN-BiLSTM-Attention, 2-class output.

Softmax output class ordering (FIXED — enforced by test_label_encoding.py):
    Index 0 = Down (class_down)  →  close < open
    Index 1 = Up   (class_up)   →  close >= open

An encoding mismatch here silently inverts all signals — spec §20 rule 1.

Architecture (per spec §7.2):
    Multi-scale Conv1D (kernels 3, 7, 15) with causal padding
    → BiLSTM
    → Optional Multi-head Self-Attention
    → Global Avg Pool + Last Timestep → Concat
    → Dense → 2-class Softmax

BiLSTM backward pass is NOT leakage (per spec §7.3):
    The backward LSTM sees later timesteps WITHIN the input window — all
    fully closed historical bars at inference time. Genuine leakage would
    come from bars AFTER the sequence end, which is prevented by causal
    padding and the direction.py guard.
"""

import os
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class CausalConv1d(nn.Module):
    """Conv1D with causal padding — mandatory per spec §7.2."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = F.pad(x, (self.padding, 0))  # causal: pad left only
        return self.conv(x)


class MultiScaleCNN(nn.Module):
    """Multi-scale Conv1D with kernels 3, 7, 15 and causal padding.

    Uses LayerNorm (not BatchNorm) for training stability at variable
    sequence lengths — correct choice for causal Conv at seq 500/750/1000.
    """

    def __init__(self, n_features: int, filters: int, dropout: float):
        super().__init__()
        # Three parallel causal conv branches
        self.conv3 = CausalConv1d(n_features, filters, kernel_size=3)
        self.conv7 = CausalConv1d(n_features, filters, kernel_size=7)
        self.conv15 = CausalConv1d(n_features, filters, kernel_size=15)

        # LayerNorm applied per-channel (filters dimension)
        self.ln3 = nn.LayerNorm(filters)
        self.ln7 = nn.LayerNorm(filters)
        self.ln15 = nn.LayerNorm(filters)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = x.permute(0, 2, 1)  # → (batch, n_features, seq_len)

        # Conv → permute → LayerNorm → GELU
        c3 = F.gelu(self.ln3(self.conv3(x).permute(0, 2, 1)))   # (batch, seq_len, filters)
        c7 = F.gelu(self.ln7(self.conv7(x).permute(0, 2, 1)))   # (batch, seq_len, filters)
        c15 = F.gelu(self.ln15(self.conv15(x).permute(0, 2, 1))) # (batch, seq_len, filters)

        # Concatenate → (batch, seq_len, filters*3)
        out = torch.cat([c3, c7, c15], dim=2)
        out = self.dropout(out)

        return out  # (batch, seq_len, filters*3)


class CNNBiLSTMAttention(nn.Module):
    """
    CNN-BiLSTM-Attention model for 2-class binary direction prediction.

    Softmax ordering: index 0 = Down, index 1 = Up.
    This is FIXED and enforced by test_label_encoding.py.

    BiLSTM backward pass note (spec §7.3):
        The backward LSTM sees later timesteps WITHIN the input window.
        These are all fully closed historical bars at inference time.
        This is NOT leakage. Genuine leakage would be bars AFTER the
        sequence end, which is prevented by causal padding and direction.py.
    """

    def __init__(
        self,
        n_features: int,
        config: Optional[dict] = None,
    ):
        super().__init__()

        if config is None:
            config = load_config()

        mc = config['model']
        self.n_features = n_features
        self.seq_len = mc['sequence_length']
        self.filters = mc['conv_filters']
        self.hidden_dim = mc['lstm_hidden_dim']
        self.n_layers = mc['lstm_layers']
        self.n_heads = mc['attention_heads']
        self.use_attention = mc['use_global_attention']
        self.dropout_rate = mc['dropout']
        self.output_classes = mc['output_classes']  # Always 2

        # Multi-scale CNN
        self.cnn = MultiScaleCNN(n_features, self.filters, self.dropout_rate)
        cnn_out_dim = self.filters * 3

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate if self.n_layers > 1 else 0,
        )
        lstm_out_dim = self.hidden_dim * 2  # bidirectional

        # Optional Multi-head Self-Attention
        # O(n²) in seq_len — profile T4 VRAM before committing (spec §7.2)
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_out_dim,
                num_heads=self.n_heads,
                dropout=self.dropout_rate,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(lstm_out_dim)

        # Output head: concat(global_avg_pool, last_timestep) → dense → softmax
        concat_dim = lstm_out_dim * 2  # pool + last
        self.head = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.output_classes),  # 2-class: [Down, Up]
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, sequence_length, n_features)

        Returns:
            (batch, 2) — softmax probabilities [p_down, p_up]
            Index 0 = Down, Index 1 = Up (FIXED)
        """
        # CNN
        cnn_out = self.cnn(x)  # (batch, seq_len, filters*3)

        # BiLSTM
        lstm_out, _ = self.lstm(cnn_out)  # (batch, seq_len, hidden*2)

        # Optional attention
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = self.attn_norm(lstm_out + attn_out)  # residual

        # Global average pool + last timestep → concat
        global_pool = lstm_out.mean(dim=1)  # (batch, hidden*2)
        last_ts = lstm_out[:, -1, :]  # (batch, hidden*2)
        concat = torch.cat([global_pool, last_ts], dim=1)  # (batch, hidden*4)

        # Dense head → softmax
        logits = self.head(concat)  # (batch, 2)
        probs = F.softmax(logits, dim=-1)  # [p_down, p_up]

        return probs


def build_model(n_features: int, config: Optional[dict] = None) -> CNNBiLSTMAttention:
    """Factory function to build the model."""
    return CNNBiLSTMAttention(n_features, config)
