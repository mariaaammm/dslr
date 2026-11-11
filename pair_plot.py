import argparse

import matplotlib.pyplot as plt
import seaborn as sns

from utils import load, COURSES, HOUSES, COLORS

# From this visualization, which features are you going to use for your logistic regression?


def pair_plot(df):
    g = sns.pairplot(df[COURSES + ["Hogwarts House"]].dropna(),
                     hue="Hogwarts House", hue_order=HOUSES, palette=COLORS,
                     diag_kind="hist", plot_kws={"s": 6, "alpha": 0.5}, height=1.4)
    g.figure.suptitle("Pair plot of Hogwarts courses", y=1.01)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    pair_plot(load(args.filename))


if __name__ == "__main__":
    main()