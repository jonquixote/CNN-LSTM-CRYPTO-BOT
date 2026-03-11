"""
models/train.py — Training loop for CNN-BiLSTM-Attention.

Per spec §7.6:
    Optimizer: AdamW, weight decay 1e-4
    LR schedule: OneCycleLR with warmup
    Loss: Focal loss (γ=2)
    Batch size: 1024
    Max epochs: 100, patience 10
    Stride: 50 bars
"""

import os
import logging
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import yaml

logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss (γ=2) for binary classification.
    Down-weights well-classified examples to focus on hard cases.
    """

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (batch, 2) softmax probabilities
            target: (batch,) integer labels 0 or 1
        """
        ce = nn.functional.cross_entropy(pred, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


# ── Dataset ───────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """
    Creates sequences of (sequence_length, n_features) for training.
    Stride = 50 bars per spec to reduce redundancy.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sequence_length: int = 750,
        stride: int = 50,
    ):
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length
        self.stride = stride

        # Generate valid sequence start indices
        self.indices = list(range(
            0,
            len(features) - sequence_length,
            stride,
        ))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.sequence_length

        X = torch.FloatTensor(self.features[start:end])
        # Label is for the bar AFTER the sequence
        y = torch.LongTensor([self.labels[end]])[0]

        return X, y


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Optional[dict] = None,
    device: str = 'auto',
    seed: int = 0,
) -> dict:
    """
    Train the model with early stopping.

    Args:
        model: CNNBiLSTMAttention model
        X_train: Training features (n_bars, n_features) — already scaled
        y_train: Training labels (n_bars,) — 0 or 1
        X_val: Validation features
        y_val: Validation labels
        config: Optional config override
        device: 'cuda', 'cpu', or 'auto'
        seed: Random seed

    Returns:
        dict with training history and best metrics
    """
    if config is None:
        config = load_config()

    mc = config['model']

    # Seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model = model.to(device)

    # Datasets
    train_ds = SequenceDataset(
        X_train, y_train,
        sequence_length=mc['sequence_length'],
        stride=50,
    )
    val_ds = SequenceDataset(
        X_val, y_val,
        sequence_length=mc['sequence_length'],
        stride=mc['sequence_length'],  # no overlap for validation
    )

    train_loader = DataLoader(
        train_ds, batch_size=mc['batch_size'],
        shuffle=True, num_workers=0, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=mc['batch_size'],
        shuffle=False, num_workers=0,
    )

    # Loss, optimizer, scheduler
    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=mc['learning_rate'],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=mc['learning_rate'],
        epochs=mc['max_epochs'],
        steps_per_epoch=len(train_loader),
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    logger.info("Training on {} (seed={}, {} train sequences, {} val sequences)".format(
        device, seed, len(train_ds), len(val_ds)
    ))

    for epoch in range(mc['max_epochs']):
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validate
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())

                # Accuracy
                predicted = pred.argmax(dim=1)
                val_correct += (predicted == y_batch).sum().item()
                val_total += y_batch.size(0)

        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        val_accuracy = val_correct / max(val_total, 1)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)

        logger.info(
            "Epoch {}/{}: train_loss={:.4f} val_loss={:.4f} val_acc={:.4f}".format(
                epoch + 1, mc['max_epochs'],
                avg_train_loss, avg_val_loss, val_accuracy
            )
        )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= mc['early_stopping_patience']:
                logger.info("Early stopping at epoch {}".format(epoch + 1))
                break

    # Restore best model
    if 'best_state' in locals():
        model.load_state_dict(best_state)

    return {
        'history': history,
        'best_val_loss': best_val_loss,
        'best_val_accuracy': max(history['val_accuracy']) if history['val_accuracy'] else 0,
        'epochs_trained': len(history['train_loss']),
        'seed': seed,
    }


def fit_scaler(X_train: np.ndarray) -> RobustScaler:
    """
    Fit RobustScaler on training data only.
    Per spec §5: RobustScaler fit only on training data per fold.
    """
    scaler = RobustScaler()
    scaler.fit(X_train)
    return scaler
