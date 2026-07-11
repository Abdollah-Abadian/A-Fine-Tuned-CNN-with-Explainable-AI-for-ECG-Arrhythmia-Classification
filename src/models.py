"""
Model architectures used in the paper: FT-CNN (dual‑branch), Standard 1D‑CNN,
ResNet, ANN, LSTM, GRU. All follow the hyperparameters given in §3.3.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input

from src.config import FT_CNN_CONFIG, NUM_CLASSES, WINDOW_SAMPLES
from src.losses import weighted_focal_categorical_crossentropy


# --------------------------------------------------------------------------- #
# FT-CNN – dual‑branch (morphology + RR features) with adaptive kernel sizing
# --------------------------------------------------------------------------- #
def build_ftcnn(
    input_length: int = WINDOW_SAMPLES,
    rr_dim: int = 4,
    num_classes: int = NUM_CLASSES,
    config=FT_CNN_CONFIG,
) -> Model:
    """Return the fully optimized FT-CNN model (Figure 4, §3.2)."""
    # Morphological branch
    morph_input = Input(shape=(input_length, 1), name="morph_input")

    # Block 1: kernel size 5, 32 filters
    x = layers.Conv1D(
        filters=config.block1_filters,
        kernel_size=config.block1_kernel,
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_lambda),
        kernel_initializer="he_normal",
        name="block1_conv"
    )(morph_input)
    x = layers.BatchNormalization(name="block1_bn")(x)
    x = layers.ReLU(name="block1_relu")(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2, name="block1_pool")(x)

    # Block 2: kernel size 3, 64 filters, spatial dropout, PReLU
    x = layers.Conv1D(
        filters=config.block2_filters,
        kernel_size=config.block2_kernel,
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_lambda),
        kernel_initializer="he_normal",
        name="block2_conv"
    )(x)
    x = layers.BatchNormalization(name="block2_bn")(x)
    x = layers.PReLU(shared_axes=[1], name="block2_prelu")(x)
    x = layers.SpatialDropout1D(rate=config.block2_spatial_dropout, name="block2_spatial_drop")(x)
    x = layers.AveragePooling1D(pool_size=2, strides=2, name="block2_pool")(x)

    # Block 3: kernel size 3, 128 filters, LeakyReLU, global average pooling
    x = layers.Conv1D(
        filters=config.block3_filters,
        kernel_size=config.block3_kernel,
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_lambda),
        kernel_initializer="he_normal",
        name="block3_conv"
    )(x)
    x = layers.BatchNormalization(name="block3_bn")(x)
    x = layers.LeakyReLU(alpha=config.leaky_relu_alpha, name="block3_leakyrelu")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)   # shape (128,)

    # RR feature branch
    rr_input = Input(shape=(rr_dim,), name="rr_input")
    rr_dense = layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(config.l2_lambda))(rr_input)

    # Fusion: concatenate
    fused = layers.Concatenate(name="fusion")([x, rr_dense])

    # Fully connected head
    y = layers.Dense(
        config.dense_1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_lambda),
        kernel_initializer="he_normal"
    )(fused)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(config.dense_2_dropout)(y)

    y = layers.Dense(
        config.dense_2,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_lambda),
        kernel_initializer="he_normal"
    )(y)
    y = layers.Dropout(config.dense_2_dropout)(y)

    y = layers.Dense(
        config.dense_3,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_lambda),
        kernel_initializer="he_normal"
    )(y)

    output = layers.Dense(num_classes, activation="softmax", name="output",
                          kernel_initializer="glorot_uniform")(y)

    model = Model(inputs=[morph_input, rr_input], outputs=output, name="FT-CNN")
    return model


# --------------------------------------------------------------------------- #
# Standard 1D-CNN (baseline, single branch)
# --------------------------------------------------------------------------- #
def build_standard_cnn(input_length: int = WINDOW_SAMPLES, num_classes: int = NUM_CLASSES) -> Model:
    """Baseline CNN with ReLU, batch norm, dropout, and standard loss (Table 7)."""
    inputs = Input(shape=(input_length, 1))
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="Standard_CNN")


# --------------------------------------------------------------------------- #
# ResNet (1D residual blocks)
# --------------------------------------------------------------------------- #
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv1D(filters, 1, strides=stride)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_resnet(input_length: int = WINDOW_SAMPLES, num_classes: int = NUM_CLASSES) -> Model:
    inputs = Input(shape=(input_length, 1))
    x = layers.Conv1D(64, 7, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(3, strides=2, padding="same")(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="ResNet")


# --------------------------------------------------------------------------- #
# ANN (multilayer perceptron)
# --------------------------------------------------------------------------- #
def build_ann(input_length: int = WINDOW_SAMPLES, num_classes: int = NUM_CLASSES) -> Model:
    inputs = Input(shape=(input_length,))
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="ANN")


# --------------------------------------------------------------------------- #
# LSTM / GRU (single branch, sequential)
# --------------------------------------------------------------------------- #
def build_lstm(input_length: int = WINDOW_SAMPLES, num_classes: int = NUM_CLASSES) -> Model:
    inputs = Input(shape=(input_length, 1))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="LSTM")

def build_gru(input_length: int = WINDOW_SAMPLES, num_classes: int = NUM_CLASSES) -> Model:
    inputs = Input(shape=(input_length, 1))
    x = layers.GRU(64, return_sequences=True)(inputs)
    x = layers.GRU(32)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="GRU")


# --------------------------------------------------------------------------- #
# Minimal baseline (no fine‑tuning) – same as standard CNN but with fixed LR,
# no focal loss, no extra regularization (used in Table 7)
# --------------------------------------------------------------------------- #
def build_minimal_baseline(input_length: int = WINDOW_SAMPLES, num_classes: int = NUM_CLASSES) -> Model:
    """No L2, no dropout, no batch norm, constant LR, standard CE (Table 7)."""
    inputs = Input(shape=(input_length, 1))
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="Minimal_Baseline")
