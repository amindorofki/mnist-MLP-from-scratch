" make predictions"

import matplotlib.pyplot as plt
import numpy as np

# import required modules
from src.activations import ReLU, Softmax
from src.data import MNISTloader
from src.layers import Dense
from src.model import Sequential


def load_trained_model(weights_path: str= 'mnist_mlp_weights.npz'):
    "loads saved model"

    print(f"[*] Loading weights  from '{weights_path}'...")
    weights = np.load(weights_path)

    num_layers = sum(1 for k in weights.files if k.startswith("W"))

    layers = []
    for i in range(num_layers):
        W = weights[f"W{i}"]
        b = weights[f"b{i}"]
        dense = Dense(in_features=W.shape[0], out_features=W.shape[1])
        dense.W = W 
        dense.b = b 
        layers.append(dense)
        if i < num_layers -1 :
            layers.append(ReLU())
    layers.append(Softmax())
        
    return Sequential(layers)

def predict(X: np.ndarray, model: Sequential):
    "runs forward pass and returns predictions with scores"

    probs = model.forward(X)
    predictions = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1) * 100.0

    return predictions, confidence, probs

def visualize_random_samples(
    X_test: np.ndarray,
    y_test_raw: np.ndarray,
    model: Sequential,
    num_samples: int =5,
):
    "selects random images from test set, predicts labels, and plots results"
    indices = np.random.choice(len(X_test), num_samples, replace= False)
    sample_X = X_test[indices]
    sample_y = y_test_raw[indices]

    # predict
    preds, confidence , _ = predict(sample_X, model)

    #Plot results
    fig, axes = plt.subplots(1, num_samples, figsize=(13,3))

    for i in range(num_samples):
        ax = axes[i]
        # reshape 784 vector back to 28 * 28 image 
        img = sample_X[i].reshape(28, 28)
        ax.imshow(img, cmap="gray")

        #green for correct otherwize red 
        is_correct = preds[i] == sample_y[i]
        color = 'green' if is_correct else "red"

        ax.set_title(
            f"Pred: {preds[i]} ({confidence[i]:.1f}%)\n True:{sample_y[i]}",
            color = color,
            fontsize=11,
            fontweight="bold",
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("predictions_sample.png", dpi=300)
    print("[+] Visual output saved 'to predictions_sample.png'")
    plt.show()


def main():
    # 1. restore trained model 
    model = load_trained_model("mnist_mlp_weights.npz")

    # 2. load test data 
    loader = MNISTloader()
    # fetch raw integer labels for easy comparison
    _, _, _, X_test, _, y_test_raw = loader.load_data()

    # 3. predict and Visualize 5 random samples 
    print("\n___ Running inference ___")
    visualize_random_samples(
        X_test, y_test_raw, model, num_samples=5
    )


if __name__ == "__main__":
    main()

