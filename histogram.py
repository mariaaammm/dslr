import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import HOUSE_COLORS, load

def homogeneity(df, course):
    means = [df.loc[df["Hogwarts House"] == h, course].mean() for h in HOUSE_COLORS]
    return (max(means) - min(means)) / df[course].std()

def histogram(df: pd.DataFrame) -> None:
    if "Hogwarts House" not in df.columns or df["Hogwarts House"].isna().all():
        raise ValueError("Error: dataset has no house labels")

    houses = df["Hogwarts House"]
    numeric_df = df.select_dtypes(include="number").drop(columns=["Index"], errors="ignore")
    courses = list(numeric_df.columns)

    if not courses:
        raise ValueError("Error: no numeric course columns found.")

    n_cols = 4
    n_rows = math.ceil(len(courses) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes_flat = list(axes.flat)

    for i, course in enumerate(courses):
        ax = axes_flat[i]
        series = numeric_df[course].dropna()
        if series.empty:
            ax.set_title(f"{course} (empty)", fontsize=9)
            continue

        bins = np.linspace(series.min(), series.max(), 31)
        for house, color in HOUSE_COLORS.items():
            mask = (houses == house) & numeric_df[course].notna()
            scores = numeric_df[course][mask].dropna()
            if not scores.empty:
                ax.hist(scores, bins=bins, alpha=0.7, color=color, label=house)

        ax.set_title(course, fontsize=9)
        ax.set_xlabel("Score", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes_flat[len(courses) :]:
        ax.set_visible(False)

    best = min(courses, key=lambda c: homogeneity(df, c))
    print(f"Most homogeneous course: {best} ({homogeneity(df, best):.3f})")

    axes_flat[0].legend(fontsize=8)
    fig.suptitle("Score distribution per course across Hogwarts houses", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    try:
        df = load(args.filename)
        histogram(df)
    except FileNotFoundError:
        print(f"Error: file not found: {args.filename}")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()