import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import textwrap

from utils import HOUSE_COLORS, COURSES, HOUSES, load


def pair_plot(df: pd.DataFrame) -> None:
    required_cols = COURSES + ["Hogwarts House"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Error: missing required columns: {', '.join(missing_cols)}")

    if df["Hogwarts House"].isna().all():
        raise ValueError("Error: dataset has no house labels")

    data = df[required_cols].dropna()
    if data.empty:
        raise ValueError("Error: no complete rows left after dropping missing values")


    hue_order = [h for h in HOUSES if h in set(data["Hogwarts House"])]
    palette = {h: HOUSE_COLORS[h] for h in hue_order if h in HOUSE_COLORS}

    g = sns.pairplot(
        data,
        hue="Hogwarts House",
        hue_order=hue_order,
        palette=palette,
        diag_kind="hist",
        height=1.6,
        aspect=1,
        plot_kws={"s": 5, "alpha": 0.5, "linewidth": 0},
        diag_kws={"element": "step", "alpha": 0.35},
    )

    for ax in g.axes.flat:
        if ax is None:
            continue
        # ax.set_xlabel(ax.get_xlabel(), fontsize=7)
        # ax.set_ylabel(ax.get_ylabel(), fontsize=7)
        ax.set_xlabel(textwrap.fill(ax.get_xlabel(), 13, break_long_words=False), fontsize=7)
        ax.set_ylabel(textwrap.fill(ax.get_ylabel(), 13, break_long_words=False), fontsize=7)
        ax.tick_params(labelsize=6)
        ax.tick_params(axis="x", labelrotation=45)

    sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 1.01), ncol=4, title=None)
    g.figure.suptitle("Pair plot of Hogwarts courses", y=1.06)
    g.figure.subplots_adjust(wspace=0.15, hspace=0.15)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    if not args.filename.endswith(".csv"):
        print("Error: file must have a .csv extension")
        return

    try:
        df = load(args.filename)
        pair_plot(df)
    except FileNotFoundError:
        print(f"Error: file not found: {args.filename}")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()