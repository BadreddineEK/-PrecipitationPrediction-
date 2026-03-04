import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn2d(input_shape=(128, 128, 4), output_size=8):
    """CNN 2D — best model: batch=32, epochs=20, RMSprop lr=0.001."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(output_size)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        loss='msle'
    )
    return model


def build_cnn3d(input_shape=(4, 128, 128, 1), output_size=8):
    """CNN 3D — captures temporal dynamics."""
    model = models.Sequential([
        layers.Conv3D(32, (2, 3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling3D((1, 2, 2)),
        layers.Conv3D(64, (2, 3, 3), activation='relu'),
        layers.MaxPooling3D((1, 2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(output_size)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        loss='msle'
    )
    return model


def build_lcnn(input_shape=(128, 128, 7), output_size=8):
    """L-CNN — 7 channels (4 original + 3 velocity/flow channels)."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(output_size)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        loss='msle'
    )
    return model
