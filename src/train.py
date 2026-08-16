"full training loop for MNIST neural network"

import time
import matplotlib.pyplot as plt
import numpy as np

# import custom build modules from src

from activations import ReLU, Softmax
from src.data import MNISTloader
from src.layers import Dense
from src.losses import Crossenthropy
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
    ax1.y_label('Loss')
    ax1.grid(true)
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


def main():
    # Hyperparameters
    EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 0.1

    print("___ 1. Loding MNIST data ___")
    # output: X:(N, 784) in range [0, 1], Y:one_hot encoded (N, 10)
    loader = MNISTloader()
    # extract one_hot encoded targets for training and testing
    X_train, Y_train, _, X_test, Y_test, _ = loader.load_data() 
    num_samples = X_train.shape[0]

    print("\n___ 2.Initializing Model Architecture ___")
    # 784 (input) -> 128 (hidden) -> 10 (output)
    dense1 = Dense(in_features = 784, out_features = 128)
    relu = ReLU()
    dense2 = Dense(in_features = 128, out_features= 10)
    softmax = Softmax()

    loss_fn = Crossenthropy()
    optimizer = SGD(lr= LEARNING_RATE)

    # history dictionary for tracking metrics 
    history = {"loss": [], "acc": []}

    print("\n___ 3. Starting training loop ___")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Shuffle dataset every epoch 
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        Y_train_shuffled = Y_train[indices]

        running_loss = 0.0
        running_acc = 0.0
        num_batches = int(np.ceil(num_samples/ BATCH_SIZE))
        
        for b in range(num_batches):
            start_idx = b *BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, num_samples)

            X_batch = X_train_shuffled[start_idx:end_idx]
            Y_batch = Y_train_shuffled[start_idx:end_idx]

            #------------ Forward Pass ------------
            Z1 = dense1.forward(X_batch)
            A1 = relu.forward(Z1)
            Z2 = dense2.forward(A1)
            A2 = softmax.forward(Z2)

            # Metrics
            batch_loss = loss_fn.forward(A2, Y_batch)
            batch_acc = compute_accuracy(A2, Y_batch)

            running_loss += batch_loss
            running_acc += batch_acc

            #------------ Backward pass ------------
            dZ2 = loss_fn.backward()
            dA1 = dense2.backward(dZ2)
            dZ1 = relu.backward(dA1)
            _ = dense1.backward(dZ1)

            #------------ Optimization ------------
            optimizer.step([dense1, dense2])

        # Record Epoch Metrics
        epoch_loss = running_loss / num_batches
        epoch_acc = running_acc / num_batches
        history["loss"].append(epoch_loss)
        history["acc"].append(epoch_acc)

    print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Loss: {epoch_loss:0.4f} | Train Acc: {epoch_acc:.2f}%")
    total_time = time.time() - strt_time 
    print(f"\n Training completed in {total_time:.2f} seconds")

    #------------ Evluation on test set ------------
    print("\n___ 4. Evluating on Test set ___")
    Z1_test = dense1.forward(X_test)
    A1_test = relu.forward(Z1_test)
    Z2_test = dense2.forward(A1_test)
    A2_test = softmax.forward(Z2_test)

    test_acc = compute_accuracy(A2_test, Y_test)
    print(f"Test set Accuracy: {test_acc:.2f}%")

    # ------------ Sqve model weights ------------
    np.savez(
        "mnist_mlp_weights.npz",
        W1=dense1.W,
        b1=dense1.b,
        W2=dense2.W,
        b2=dense2.b,
    )

    print("[+] Model weights saved to 'mnist_mlp_weights.npz'")

    #------- plot metrics

    plot_metrics(history)

if __name__ == "__main__":
    main()

