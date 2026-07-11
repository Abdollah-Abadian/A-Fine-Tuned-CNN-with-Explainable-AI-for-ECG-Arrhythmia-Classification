"""
Training utilities: compile and fit models with cosine annealing, early stopping,
and model checkpointing.
"""

import os
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger,
)
from tensorflow.keras.optimizers import Adam

from src.config import TRAIN_CONFIG
from src.losses import weighted_focal_categorical_crossentropy, compute_class_weights


class CosineAnnealingWarmRestarts(tf.keras.callbacks.Callback):
    """Cosine annealing learning rate scheduler with warm restarts."""
    def __init__(self, initial_lr=1e-3, min_lr=1e-6, cycle_epochs=50, verbose=0):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.cycle_epochs = cycle_epochs
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.cycle_epochs == 0:
            # restart cycle
            current_cycle = epoch // self.cycle_epochs
            lr = self.initial_lr / (2 ** current_cycle) if current_cycle > 0 else self.initial_lr
        else:
            cycle_progress = (epoch % self.cycle_epochs) / self.cycle_epochs
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + tf.cos(tf.constant(cycle_progress * 3.14159)))
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        if self.verbose:
            print(f"Epoch {epoch+1}: Learning rate = {lr:.2e}")


def train_ftcnn(
    X_train, rr_train, y_train,
    X_val, rr_val, y_val,
    batch_size=TRAIN_CONFIG.batch_size,
    max_epochs=TRAIN_CONFIG.max_epochs,
    patience=TRAIN_CONFIG.early_stopping_patience,
    model_dir="results/models",
    verbose=1,
):
    """Train the dual‑branch FT‑CNN with focal loss and cosine annealing."""
    os.makedirs(model_dir, exist_ok=True)
    model = build_ftcnn()
    class_weights = compute_class_weights(y_train, NUM_CLASSES)
    loss_fn = weighted_focal_categorical_crossentropy(class_weights)

    model.compile(
        optimizer=Adam(learning_rate=TRAIN_CONFIG.initial_lr,
                       beta_1=TRAIN_CONFIG.adam_beta_1,
                       beta_2=TRAIN_CONFIG.adam_beta_2,
                       epsilon=TRAIN_CONFIG.adam_epsilon),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    callbacks = [
        CosineAnnealingWarmRestarts(
            initial_lr=TRAIN_CONFIG.initial_lr,
            min_lr=TRAIN_CONFIG.min_lr,
            cycle_epochs=TRAIN_CONFIG.cosine_annealing_cycle_epochs,
            verbose=0
        ),
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ModelCheckpoint(os.path.join(model_dir, "ftcnn_best.h5"),
                        monitor="val_loss", save_best_only=True),
        CSVLogger(os.path.join(model_dir, "ftcnn_training_log.csv")),
    ]

    history = model.fit(
        [X_train, rr_train], y_train,
        validation_data=([X_val, rr_val], y_val),
        batch_size=batch_size,
        epochs=max_epochs,
        callbacks=callbacks,
        verbose=verbose,
    )
    return model, history


def train_single_input_model(
    model_builder,
    X_train, y_train,
    X_val, y_val,
    batch_size=TRAIN_CONFIG.batch_size,
    max_epochs=TRAIN_CONFIG.benchmark_epochs,
    patience=TRAIN_CONFIG.benchmark_early_stopping_patience,
    model_dir="results/models",
    model_name="model",
    use_class_weights=True,
    verbose=1,
):
    """Train a single‑input model (Standard CNN, ResNet, ANN, LSTM, GRU)."""
    os.makedirs(model_dir, exist_ok=True)
    model = model_builder()
    loss = "categorical_crossentropy"
    class_weight_dict = None
    if use_class_weights:
        # compute class weights for the training set
        class_weights = compute_class_weights(y_train, NUM_CLASSES)
        class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
        # Keras class_weight expects dict
    model.compile(optimizer=Adam(learning_rate=TRAIN_CONFIG.initial_lr),
                  loss=loss, metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ModelCheckpoint(os.path.join(model_dir, f"{model_name}_best.h5"),
                        monitor="val_loss", save_best_only=True),
        CSVLogger(os.path.join(model_dir, f"{model_name}_training_log.csv")),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=max_epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=verbose,
    )
    return model, history
