import argparse
import json
import numpy as np

from utils import load, COURSES, HOUSES, StandardScaler


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.1, iterations=500):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = 0.0
        self.cost_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def cost(self, h, y):
        h = np.clip(h, 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0
        self.cost_history = []

        for _ in range(self.iterations):
            h = self.sigmoid(X @ self.weights + self.bias)
            self.weights -= self.lr * (X.T @ (h - y)) / m
            self.bias -= self.lr * np.sum(h - y) / m
            self.cost_history.append(self.cost(h, y))

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

    @classmethod
    def from_dict(cls, d):
        model = cls()
        model.weights = np.array(d["weights"])
        model.bias = d["bias"]
        return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()

    df = load(args.dataset)
    features = COURSES

    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models_data = {}

    for house in HOUSES:
        y = (df["Hogwarts House"] == house).astype(int).values
        
        model = LogisticRegressionScratch(learning_rate=0.1, iterations=1000)
        model.fit(X_scaled, y)
        
        models_data[house] = model.to_dict()
    
    output_data = {
        "features": features,
        "scaler_mean": scaler.mean.tolist(),
        "scaler_std": scaler.std.tolist(),
        "models": models_data
    }

    with open("weights.json", "w") as f:
        json.dump(output_data, f, indent=4)
        
    print("Training complete! Weights and scaler saved to weights.json")

if __name__ == "__main__":
    main()