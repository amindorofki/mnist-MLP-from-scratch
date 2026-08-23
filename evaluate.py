"detailed evaluation: confusion matrix, per-class accuracy, misclassified samples"

import numpy as np
import matplotlib.pyplot as plt

from src.data import MNISTloader
from predict import load_trained_model, predict


def confusion_matrix(y_true, y_pred, num_classes=10):
    "builds an NxN matrix: rows=true label, cols=predicted label"
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_accuracy(cm):
    "diagonal / row sum, per class"
    return np.diag(cm) / cm.sum(axis=1)


def plot_confusion_matrix(cm, path="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    print(f"[+] Saved to '{path}'")


def visualize_misclassified(X_test, y_true, y_pred, probs, num_samples=8, path="misclassified.png"):
    "shows examples the model got wrong, with true label vs prediction"
    wrong_idx = np.where(y_true != y_pred)[0]

    if len(wrong_idx) == 0:
        print("[+] No misclassified samples found!")
        return

    n = min(num_samples, len(wrong_idx))
    chosen = np.random.choice(wrong_idx, n, replace=False)

    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, chosen):
        img = X_test[idx].reshape(28, 28)
        confidence = np.max(probs[idx]) * 100
        ax.imshow(img, cmap="gray")
        ax.set_title(f"True:{y_true[idx]} Pred:{y_pred[idx]}\n({confidence:.1f}%)",
                     color="red", fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    print(f"[+] Saved {n} misclassified samples to '{path}'")


def main():
    model = load_trained_model("mnist_mlp_weights.npz")
    loader = MNISTloader()
    _, _, _, X_test, _, y_test_raw = loader.load_data()

    preds, confidence, probs = predict(X_test, model)
    cm = confusion_matrix(y_test_raw, preds)

    print("=== Per-class accuracy ===")
    for i, acc in enumerate(per_class_accuracy(cm)):
        print(f"Digit {i}: {acc*100:.2f}%")

    plot_confusion_matrix(cm)
    visualize_misclassified(X_test, y_test_raw, preds, probs)


if __name__ == "__main__":
    main()