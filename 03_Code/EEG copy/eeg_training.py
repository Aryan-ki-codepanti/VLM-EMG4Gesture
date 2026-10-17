from pathlib import Path
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def make_loader(dataset, batch_size=32, shuffle=False, num_workers=0):
    """Create a DataLoader for an EEGDataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    """
    Run one training or evaluation epoch.

    If optimizer is provided, update model parameters.
    Returns mean loss and accuracy.
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        x, y = batch[0], batch[1]

        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.long).view(-1)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("DataLoader produced no samples.")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "samples": total_samples,
    }


def fit_model(
    model,
    train_dataset,
    val_dataset,
    device,
    max_epochs=50,
    batch_size=32,
    learning_rate=3e-4,
    weight_decay=1e-4,
    patience=8,
    checkpoint_path=None,
    seed=42,
):
    """
    Train using validation loss for early stopping.

    The best model checkpoint is restored before returning.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = make_loader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = make_loader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        })

        print(
            f"Epoch {epoch:02d}/{max_epochs} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"train acc {train_metrics['accuracy']:.3f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val acc {val_metrics['accuracy']:.3f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            if checkpoint_path is not None:
                path = Path(checkpoint_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": best_state,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                }, path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


@torch.no_grad()
def predict_model(model, dataset, device, batch_size=32):
    """Return true labels, predicted labels and class probabilities."""
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    all_y = []
    all_pred = []
    all_probs = []

    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device, dtype=torch.float32)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        all_y.extend(y.cpu().numpy().astype(int).tolist())
        all_pred.extend(logits.argmax(
            dim=1).cpu().numpy().astype(int).tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    return (
        np.asarray(all_y),
        np.asarray(all_pred),
        np.asarray(all_probs),
    )
