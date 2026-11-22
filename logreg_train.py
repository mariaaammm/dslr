import argparse
import json
import numpy as np
import sys

from utils import load, HOUSES, StandardScaler, SELECTED_FEATURES


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.1, iterations=500):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = 0.0
        # self.cost_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def cost(self, h, y):
        h = np.clip(h, 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))

    def fit(self, X, y, batch_size=None):
        """batch_size=None -> batch GD (whole dataset per update), 1 -> SGD, k -> mini-batch GD."""
        m, n = X.shape
        if batch_size is None or batch_size > m:
            batch_size = m
        self.weights = np.zeros(n)
        self.bias = 0.0
        # self.cost_history = []
        rng = np.random.default_rng(42)

        for _ in range(self.iterations):
            order = rng.permutation(m) # new order every epoch
            for start in range(0, m, batch_size):
                idx = order[start:start + batch_size]
                Xb, yb = X[idx], y[idx]
                h = self.sigmoid(Xb @ self.weights + self.bias)
                self.weights -= self.lr * (Xb.T @ (h - yb)) / len(yb)
                self.bias -= self.lr * np.sum(h - yb) / len(yb)
            # one cost value per epoch, measured on the whole dataset
            # self.cost_history.append(self.cost(self.sigmoid(X @ self.weights + self.bias), y))

        return self

    def predict_proba(self, X):
        return self.sigmoid(X @ self.weights + self.bias)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def to_dict(self):
        return {
            "weights": self.weights.tolist() if self.weights is not None else [],
            "bias": self.bias
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--method", choices=["batch", "sgd", "minibatch"], default="batch")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("-o", "--output", default="weights.json")
    args = parser.parse_args()

    batch_size = {"batch": None, "sgd": 1, "minibatch": args.batch_size}[args.method]
    epochs = args.epochs if args.epochs is not None else {"batch": 1000, "sgd": 30, "minibatch": 200}[args.method]

    df = load(args.dataset)
    features = SELECTED_FEATURES

    if epochs < 1 or (batch_size is not None and batch_size < 1):
        sys.exit("Error: --epochs and --batch-size must be positive")
    if df["Hogwarts House"].isna().all():
        sys.exit("Error: no house labels (use dataset_train.csv)")
    missing = [c for c in features if c not in df.columns]
    if missing:
        sys.exit(f"Error: missing columns: {', '.join(missing)}")

    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models_data = {}

    for house in HOUSES:
        y = (df["Hogwarts House"] == house).astype(int).values
        
        model = LogisticRegressionScratch(learning_rate=0.1, iterations=epochs)
        model.fit(X_scaled, y, batch_size=batch_size)
        
        models_data[house] = model.to_dict()
    
    output_data = {
        "features": features,
        "scaler_mean": scaler.mean.tolist(),
        "scaler_std": scaler.std.tolist(),
        "models": models_data
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Training complete! Weights and scaler saved to {args.output}")

if __name__ == "__main__":
    main()