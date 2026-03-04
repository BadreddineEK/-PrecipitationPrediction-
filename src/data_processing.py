import numpy as np
import pandas as pd
import os


def load_npz_files(folder_path, n_files=1000):
    """
    Load n_files .npz files from folder_path.
    Returns X of shape (n_files, 128, 128, 4).
    """
    files = sorted(os.listdir(folder_path))[:n_files]
    X = []
    for f in files:
        if f.endswith(".npz"):
            data = np.load(os.path.join(folder_path, f))
            arr = list(data.values())[0]  # shape (4, 128, 128)
            X.append(arr.transpose(1, 2, 0))  # -> (128, 128, 4)
    return np.array(X)


def load_ytrain(csv_path, n_files=1000):
    """
    Load ytrain.csv and group by file (8 values per file).
    Returns y of shape (n_files, 8).
    """
    df = pd.read_csv(csv_path, header=None)
    values = df.values.flatten()
    n = n_files * 8
    y = values[:n].reshape(n_files, 8)
    return y


def log_transform(data):
    """Apply log(x + 1) transformation."""
    return np.log1p(data)


def inverse_log_transform(data):
    """Apply exp(x) - 1 inverse transformation."""
    return np.expm1(data)


def reshape_for_cnn2d(X):
    """X shape: (n, 128, 128, 4) — ready for CNN 2D."""
    return X


def reshape_for_cnn3d(X):
    """X shape: (n, 4, 128, 128, 1) — ready for CNN 3D."""
    return X.transpose(0, 3, 1, 2)[:, :, :, :, np.newaxis]


def msle(y_true, y_pred):
    """Mean Squared Logarithmic Error."""
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    return np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)
