import argparse
import matplotlib.pyplot as plt
import numpy as np

from utils import load, COURSES, HOUSES

# Which Hogwarts course has a homogeneous score distribution between all four houses?




# def homogeneity(df, course):
#     means = [df.loc[df["Hogwarts House"] == h, course].mean() for h in HOUSES]
#     return (max(means) - min(means)) / df[course].std()

def histogram(df):
    n_cols = 4
    n_rows = int(np.ceil(len(COURSES) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))

    for ax, course in zip(axes.flat, COURSES):
        for h in HOUSES:
            scores = df.loc[df["Hogwarts House"] == h, course].dropna()
            ax.hist(scores, alpha=0.5, label=h)
        ax.set_title(course)
    for ax in list(axes.flat)[len(COURSES):]:
        ax.axis("off")
    axes.flat[0].legend()

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    histogram(load(args.filename))


if __name__ == "__main__":
    main()