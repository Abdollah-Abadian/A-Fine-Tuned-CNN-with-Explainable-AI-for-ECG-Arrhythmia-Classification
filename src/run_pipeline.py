#!/usr/bin/env python
"""
Master script to run the entire pipeline: download, preprocess, train,
evaluate, ablation, robustness, XAI.
"""

import os
import argparse
import json
import numpy as np
import tensorflow as tf

from src.config import RANDOM_SEED
from src.download_mitbih import download as download_mitbih
from src.preprocessing import run_pipeline as preprocess_pipeline, synthesize_sample_csv
from src.dataset import load_split, to_one_hot
from src.train import train_ftcnn, train_single_input_model
from src.models import (
    build_ftcnn,
    build_standard_cnn,
    build_resnet,
    build_ann,
    build_lstm,
    build_gru,
    build_minimal_baseline,
)
from src.evaluate import compute_metrics, per_class_metrics, bootstrap_confidence_interval
from src.ablation import run_ablation_hyperparameters, run_kernel_ablation
from src.robustness import apply_robustness_pipeline
from src.xai import get_gradcam_heatmap, integrated_gradients
from src.utils import set_seed, plot_confusion_matrix, plot_training_history
from src.losses import compute_class_weights

def main(args):
    set_seed(RANDOM_SEED)

    if args.download:
        download_mitbih("data/raw")

    if args.preprocess:
        preprocess_pipeline("data/raw", "data/processed")
        synthesize_sample_csv("data/sample/sample_beats.csv")

    # Load data
    train = load_split("data/processed/train.npz")
    val = load_split("data/processed/val.npz")
    test = load_split("data/processed/test.npz")
    X_train, rr_train, y_train = train["X"], train["rr"], train["y"]
    X_val, rr_val, y_val = val["X"], val["rr"], val["y"]
    X_test, rr_test, y_test = test["X"], test["rr"], test["y"]

    # Convert labels to one-hot for training
    y_train_oh = to_one_hot(y_train)
    y_val_oh = to_one_hot(y_val)
    y_test_oh = to_one_hot(y_test)

    if args.train:
        # Train FT-CNN
        print("Training FT-CNN...")
        ftcnn_model, ftcnn_history = train_ftcnn(
            X_train, rr_train, y_train_oh,
            X_val, rr_val, y_val_oh,
            model_dir="results/models"
        )
        # Save history plot
        plot_training_history(ftcnn_history, save_path="results/figures/ftcnn_training.png")

        # Train baselines (only if requested)
        if args.baselines:
            models = {
                "Standard_CNN": build_standard_cnn,
                "ResNet": build_resnet,
                "ANN": build_ann,
                "LSTM": build_lstm,
                "GRU": build_gru,
                "Minimal_Baseline": build_minimal_baseline,
            }
            for name, builder in models.items():
                print(f"Training {name}...")
                model, hist = train_single_input_model(
                    builder,
                    X_train, y_train_oh,
                    X_val, y_val_oh,
                    model_dir="results/models",
                    model_name=name,
                )
                plot_training_history(hist, save_path=f"results/figures/{name}_training.png")

    if args.evaluate:
        # Load best FT-CNN
        ftcnn = tf.keras.models.load_model("results/models/ftcnn_best.h5")
        pred_probs = ftcnn.predict([X_test, rr_test])
        y_pred = pred_probs.argmax(axis=-1)
        metrics = compute_metrics(y_test, y_pred)
        print("FT-CNN test metrics:", json.dumps(metrics, indent=2))
        per_class = per_class_metrics(y_test, y_pred, labels=[0,1,2,3,4])
        print("Per-class metrics:", json.dumps(per_class, indent=2))
        # Bootstrap CI for accuracy
        ci_low, ci_high, _ = bootstrap_confidence_interval(y_test, y_pred, metric_func=lambda a,b: np.mean(a==b))
        print(f"95% CI accuracy: [{ci_low:.4f}, {ci_high:.4f}]")
        # Confusion matrix
        plot_confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4],
                              title="FT-CNN Confusion Matrix",
                              save_path="results/figures/ftcnn_cm.png")

    if args.ablation:
        # Run hyperparameter ablations
        print("Running hyperparameter ablations...")
        ab_results = run_ablation_hyperparameters(X_train, rr_train, y_train, X_val, rr_val, y_val, X_test, rr_test, y_test)
        # Save to results/ablation
        os.makedirs("results/ablation", exist_ok=True)
        with open("results/ablation/hyperparam_ablation.json", "w") as f:
            json.dump(ab_results, f, indent=2)

    if args.robustness:
        # Apply perturbations and evaluate
        print("Running robustness tests...")
        # For a subset of test samples
        idx = np.random.choice(len(X_test), 100, replace=False)
        X_sub = X_test[idx]
        rr_sub = rr_test[idx]
        y_sub = y_test[idx]
        conditions = ["snr20", "snr10", "snr5", "amp_scale", "time_shift"]
        results = {}
        ftcnn = tf.keras.models.load_model("results/models/ftcnn_best.h5")
        for cond in conditions:
            perturbed = np.array([apply_robustness_pipeline(x.squeeze(), cond) for x in X_sub])[..., np.newaxis]
            preds = ftcnn.predict([perturbed, rr_sub]).argmax(axis=-1)
            acc = np.mean(preds == y_sub)
            results[cond] = acc
        print("Robustness results:", json.dumps(results, indent=2))

    if args.xai:
        # Compute XAI metrics
        print("Running XAI analysis...")
        ftcnn = tf.keras.models.load_model("results/models/ftcnn_best.h5")
        # Select a few samples
        idx = np.random.choice(len(X_test), 5, replace=False)
        for i in idx:
            x = X_test[i][np.newaxis, ...]
            rr = rr_test[i][np.newaxis, ...]
            true_label = y_test[i]
            # Grad-CAM heatmap
            heatmap = get_gradcam_heatmap(ftcnn, x, rr, true_label)
            # Integrated Gradients
            ig = integrated_gradients(ftcnn, x, rr, true_label)
            # Faithfulness metrics (placeholder)
            # Store results
            print(f"Sample {i}: heatmap shape {heatmap.shape}, IG shape {ig.shape}")

    print("Pipeline completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full ECG classification pipeline.")
    parser.add_argument("--download", action="store_true", help="Download MIT-BIH dataset")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing and create splits")
    parser.add_argument("--train", action="store_true", help="Train FT-CNN and baselines")
    parser.add_argument("--baselines", action="store_true", help="Train baseline models (if --train)")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate models on test set")
    parser.add_argument("--ablation", action="store_true", help="Run ablation studies")
    parser.add_argument("--robustness", action="store_true", help="Run robustness tests")
    parser.add_argument("--xai", action="store_true", help="Run XAI analysis")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    args = parser.parse_args()

    if args.all:
        args.download = args.preprocess = args.train = args.baselines = args.evaluate = args.ablation = args.robustness = args.xai = True
    main(args)
