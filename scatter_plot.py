import argparse

import matplotlib.pyplot as plt
import numpy as np

from utils import load, COURSES, HOUSES, COLORS

# What are the two features that are similar?


def most_similar_pair(df):
    """Return ((a, b), r) for the pair of courses with the largest |Pearson r|."""
    best, best_r = None, 0.0
    for i, a in enumerate(COURSES):
        for b in COURSES[i + 1:]:                  # each pair once
            pair = df[[a, b]].dropna()             # rows where both scores exist
            r = np.corrcoef(pair[a], pair[b])[0, 1]
            if abs(r) > abs(best_r):
                best, best_r = (a, b), r
    return best, best_r


def scatter_plot(df):
    (a, b), r = most_similar_pair(df)
    print(f"Most similar features: {a} / {b} (r = {r:.4f})")

    plt.figure(figsize=(8, 6))
    for h in HOUSES:
        d = df[df["Hogwarts House"] == h]
        plt.scatter(d[a], d[b], s=14, alpha=0.6, color=COLORS[h], label=h)
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
    scatter_plot(load(args.filename))


if __name__ == "__main__":
    main()