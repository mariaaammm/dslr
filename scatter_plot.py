import argparse

import matplotlib.pyplot as plt
import numpy as np

from utils import load, COURSES, HOUSES, HOUSE_COLORS

def pearson_corr(x, y):
    dx = x - np.mean(x)
    dy = y - np.mean(y)
    denom = np.sqrt(np.sum(dx ** 2) * np.sum(dy ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum(dx * dy) / denom)

# What are the two features that are similar?
def most_similar_pair(df):
    best, best_r = None, 0.0

    for i, a in enumerate(COURSES):
        for b in COURSES[i + 1:]:
            pair = df[[a, b]].dropna()
            if len(pair) < 2:
                continue
            r = pearson_corr(pair[a].to_numpy(), pair[b].to_numpy())
            if abs(r) > abs(best_r):
                best, best_r = (a, b), r

    return best, best_r


def scatter_plot(df):
    (a, b), r = most_similar_pair(df)
    print(f"Most similar features: {a} / {b} (r = {r:.4f})")

    plt.figure(figsize=(8, 6))
    for h in HOUSES:
        d = df[df["Hogwarts House"] == h]
        plt.scatter(d[a], d[b], s=14, alpha=0.6, color=HOUSE_COLORS[h], label=h)
    plt.xlabel(a)
    plt.ylabel(b)
    plt.title(f"{a} vs {b} (r = {r:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    try:
        df = load(args.filename)
        scatter_plot(df)
    except FileNotFoundError:
            print(f"Error: file not found: {args.filename}")
    except Exception as e:
            print(e)


if __name__ == "__main__":
    main()