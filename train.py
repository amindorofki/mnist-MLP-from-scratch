"full training loop for MNIST neural network"

import argparse
import time
import matplotlib.pyplot as plt
import numpy as np

# import custom build modules from src

from src.activations import ReLU, Softmax
from src.data import MNISTloader, train_val_split
from src.layers import Dense
from src.losses import CrossEntropy
from src.model import Sequential
from src.optimizers import SGD

def compute_accuracy(y_pred_probs: np.ndarray, y_true: np.ndarray) -> float:
    "compute classification accuracy"
    predictions = np.argmax(y_pred_probs, axis=1)
    labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
    return np.mean (predictions == labels) * 100.0

def plot_metrics(history: dict):
    "Plots and saves training loss and accuracy curves"
    epochs = range(1, len(history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Loss Plot
    ax1.plot(epochs, history["loss"], "b-o", label = "Train Loss")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    #Accuracy plot
    ax2.plot(epochs, history["acc"], "g-o", label = "Train Accuracy")
    ax2.set_title("Training Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi = 300)
    print("\n[+] Learning curves saved as 'learning_curves.png'")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Train an MLP on MNIST")
    parser.add_argument("--epochs", type=int, default =10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128],
                        help="size of hidden layers, e.g. hidden-dims 256 128")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--l2", type =float, default=0.0)
    return parser.parse_args()

def build_model(input_dim, hidden_dims, output_dim):
    "builds a sequential MLP with an arbitrary number of hidden layers"
    layers = []
    dims = [input_dim] + hidden_dims + [output_dim]
    for i in range(len(dims)- 1):
        layers.append(Dense(in_features=dims[i], out_features=dims[i+1]))
        if i < len(dims) -2 :
            layers.append(ReLU())
    layers.append(Softmax())
    return Sequential(layers)

def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("___ 1.Loading MNIST data ___")
    loader = MNISTloader()
    X_train, Y_train, _, X_test, Y_test, _ = loader.load_data()
    X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, val_ratio = 0.1)
    num_samples = X_train.shape[0]

    print("\n___ 2. Initializing Model Architecture ___")
    model = build_model(input_dim=784, hidden_dims=args.hidden_dims, output_dim=10)
    
    loss_fn = CrossEntropy()
    optimizer =SGD(lr=args.lr)

    history = {"loss": [], "acc": []}

    print("\n___ 3. Starting training loop ___")
    start_time = time.time()

    for epoch in range(1, args.epochs +1):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_shuf, Y_shuf = X_train[indices], Y_train[indices]

        running_loss, running_acc = 0.0, 0.0
        num_batches = int(np.ceil(num_samples / args.batch_size))

        for b in range(num_batches):
            s, e = b * args.batch_size, min((b + 1) * args.batch_size, num_samples)
            X_batch, Y_batch = X_shuf[s:e], Y_shuf[s:e]

            probs = model.forward(X_batch)
            loss = loss_fn.forward(probs, Y_batch)
            acc = compute_accuracy(probs, Y_batch)

            running_loss += loss
            running_acc += acc

            dout = loss_fn.backward()
            model.backward(dout, l2_lambda = args.l2 )
            optimizer.step(model.trainable_layers())
        
        epoch_loss = running_loss / num_batches
        epoch_acc = running_acc / num_batches
        history["loss"].append(epoch_loss)
        history["acc"].append(epoch_acc)
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {epoch_loss:.4f} | Train acc: {epoch_acc:.2f}%")
        val_probs = model.forward(X_val)
        val_acc = compute_accuracy(val_probs, Y_val)
        print(f"Val acc: {val_acc:.2f}%")
        
    print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")

    print("\n___ 4. Evaluating on Test set ___")
    test_probs = model.forward(X_test)
    test_acc = compute_accuracy(test_probs, Y_test)
    print(f"Test set Accuracy: {test_acc:.2f}%")
    
    weights = {}
    for i, layer in enumerate(model.trainable_layers()):
        weights[f"W{i}"] = layer.W 
        weights[f"b{i}"] = layer.b 
    np.savez("mnist_mlp_weights.npz", **weights)
    print("[+] Model weights saved to 'mnist_mlp_weights.npz'")

    plot_metrics(history)

if __name__ == "__main__":
    main()

