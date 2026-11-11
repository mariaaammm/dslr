import argparse
import json,sys
import numpy as np
import pandas as pd

from utils import load, HOUSES, sigmoid

def fill_nan_with_mean(df, features, means):
    for i, f in enumerate(features):
        df[f] = df[f].fillna(means[i])
    return df

def main():
    parser = argparse.ArgumentParser(description="Predict the Hogwarts house of every student")
    parser.add_argument("dataset")
    parser.add_argument("weights")
    args = parser.parse_args()

    try:
        with open(args.weights) as f:
            model = json.load(f)
    except FileNotFoundError:
        sys.exit(f"Error: file not found: {args.weights}")

    features = model["features"]
    scaler_mean = np.array(model["scaler_mean"])
    scaler_std = np.array(model["scaler_std"])
    models = model["models"]

    df = load(args.dataset)
    df_clean = fill_nan_with_mean(df.copy(), features, scaler_mean)

    X = df_clean[features].values
    X_scaled = (X - scaler_mean) / scaler_std

    probabilities = {}
    for house in HOUSES:
        w = np.array(models[house]["weights"])
        b = models[house]["bias"]
        probabilities[house] = sigmoid(X_scaled @ w + b)

    prob_matrix = np.column_stack([probabilities[h] for h in HOUSES])

    best_house_indices = np.argmax(prob_matrix, axis=1)
    predicted_houses = [HOUSES[i] for i in best_house_indices]

    output = pd.DataFrame({
        "Index": df["Index"] if "Index" in df.columns else range(len(df)),
        "Hogwarts House": predicted_houses
    })
    
    output.to_csv("houses.csv", index=False)
    print("Predictions complete! Results saved to houses.csv")

if __name__ == "__main__":
    main()