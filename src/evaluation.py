import numpy as np
import matplotlib.pyplot as plt


def msle_score(y_true, y_pred):
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    return np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)


def plot_predictions(y_true, y_pred, n_samples=5, title="Predictions vs Ground Truth"):
    """
    Plot predicted vs real precipitation for n_samples.
    """
    fig, axes = plt.subplots(n_samples, 1, figsize=(10, 3 * n_samples))
    for i in range(n_samples):
        ax = axes[i] if n_samples > 1 else axes
        ax.plot(y_true[i], label='Ground Truth', marker='o', color='steelblue')
        ax.plot(y_pred[i], label='Prediction', marker='x', linestyle='--', color='tomato')
        ax.set_title(f"Sample {i + 1} — MSLE: {msle_score(y_true[i:i+1], y_pred[i:i+1]):.4f}")
        ax.set_xlabel("Timestep (6 min intervals)")
        ax.set_ylabel("Precipitation (log scale)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


def plot_training_history(history, title="Training History"):
    """
    Plot training loss over epochs from a Keras history object.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history['loss'], label='Train Loss', color='steelblue')
    if 'val_loss' in history.history:
        ax.plot(history.history['val_loss'], label='Val Loss', color='tomato', linestyle='--')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSLE Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
