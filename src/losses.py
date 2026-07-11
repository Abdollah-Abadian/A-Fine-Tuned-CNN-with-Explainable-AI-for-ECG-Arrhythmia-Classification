"""
Custom loss function used by the FT-CNN: weighted categorical cross-entropy
combined with a focal-loss modulation term (gamma = 2.0), as described in
§3.2/§3.3 of the paper.

    L = - sum_c  w_c * (1 - p_c)^gamma * y_c * log(p_c)

where w_c is the inverse-frequency class weight and p_c is the predicted
softmax probability for class c.
"""

import numpy as np
import tensorflow as tf

from src.config import FOCAL_LOSS_GAMMA


def compute_class_weights(y_integer_labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Inverse-frequency class weights, normalized to mean 1.0."""
    counts = np.bincount(y_integer_labels, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1, counts)
    weights = counts.sum() / (num_classes * counts)
    return weights.astype(np.float32)


def weighted_focal_categorical_crossentropy(
    class_weights: np.ndarray, gamma: float = FOCAL_LOSS_GAMMA, epsilon: float = 1e-7
):
    """Returns a Keras-compatible loss function closing over fixed class
    weights and focal gamma."""
    class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_modulation = tf.pow(1.0 - y_pred, gamma)
        weighted = class_weights_tensor * focal_modulation * cross_entropy
        return tf.reduce_sum(weighted, axis=-1)

    loss_fn.__name__ = "weighted_focal_categorical_crossentropy"
    return loss_fn


def standard_categorical_crossentropy():
    """Plain CE, used in ablation Table 9/§4.3 ('standard categorical
    cross-entropy without class weights')."""
    def loss_fn(y_true, y_pred, epsilon: float = 1e-7):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        return -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

    loss_fn.__name__ = "standard_categorical_crossentropy"
    return loss_fn


def weighted_categorical_crossentropy_no_focal(class_weights: np.ndarray, epsilon: float = 1e-7):
    """Weighted CE without the focal modulation term — used in the ablation
    study to isolate the contribution of the focal component (§4.3)."""
    class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weighted = class_weights_tensor * cross_entropy
        return tf.reduce_sum(weighted, axis=-1)

    loss_fn.__name__ = "weighted_categorical_crossentropy_no_focal"
    return loss_fn
