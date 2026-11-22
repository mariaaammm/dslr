import sys
import pandas as pd
import numpy as np

HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

COURSES = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination",
           "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions",
           "Care of Magical Creatures", "Charms", "Flying"]

SELECTED_FEATURES = ["Herbology", "Defense Against the Dark Arts", "Divination",
           "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions",
           "Charms", "Flying"]

HOUSE_COLORS = {
    "Gryffindor": "#5e030a",
    "Hufflepuff": "#ecce0b",
    "Ravenclaw": "#12013f",
    "Slytherin": "#012910",
}

def load(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: file not found: {path}")
        sys.exit(1)

class StandardScaler:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, X):
        X = np.asarray(X, dtype = np.float64)

        self.mean = np.nanmean(X, axis=0)   # NaN (missing scores) are ignored
        self.std = np.nanstd(X, axis=0)

        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        X = np.where(np.isnan(X), self.mean, X)
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))