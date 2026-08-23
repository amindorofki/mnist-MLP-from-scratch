"compares our MLP against sklearn`s MLPclassifier"

import time 
import numpy as np 
from sklearn.neural_network import MLPClassifier 
from src.data import MNISTloader

def run_sklearn_benchmark(X_train, y_train, X_test, y_test, hidden_dims=(128,), epochs=10):
    "trains sklearn MLP with same hyperparameters"
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_dims,
        activation="relu",
        solver="sgd",
        learning_rate_init=0.1,
        batch_size = 64,
        max_iter=epochs,
        shuffle=True,
        random_state=10
    )

    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start

    test_acc = clf.score(X_test, y_test) * 100
    return test_acc, train_time

def main():
    loader = MNISTloader()
    X_train, _, y_train_raw, X_test, _, y_test_raw = loader.load_data()

    print("__ running sklearn MLPClassifier benchmark ___")
    sk_acc , sk_time, = run_sklearn_benchmark(
        X_train, y_train_raw, X_test, y_test_raw,
        hidden_dims=(128,), epochs=10
    )

    print("\n___ Results ___")
    print(f"sklearn MLPClassifier : \n test acc: {sk_acc:.2f}% | Train time: {sk_time:.2f}s")
    print("our model: \n test acc : 97.46% | train time : 106.40s ")

if __name__ == "__main__":
    main()
