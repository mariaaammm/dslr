import math
import sys
import argparse
import pandas as pd
import numpy as np

from utils import load

def ft_count(v):
    count = 0
    for _ in v:
        count += 1
    return count


def ft_sum(v):
    total = 0.0
    for x in v:
        total += x
    return total


def ft_mean(v):
    return ft_sum(v) / len(v)


def ft_var(v):
    if len(v) < 2:
        raise ValueError("Variance requires at least two values")
    m = ft_mean(v)
    return ft_sum((x - m) ** 2 for x in v) / (len(v) - 1)

def ft_std(v):
    return math.sqrt(ft_var(v))

# def ft_min(v): return reduce(lambda a, b: b if b < a else a, v)
# def ft_max(v): return reduce(lambda a, b: b if b > a else a, v)

def ft_min(v):
    my_min = v[0]
    for x in v[1:]:
        if x < my_min:
            my_min = x
    return my_min


def ft_max(v):
    my_max = v[0]
    for x in v[1:]:
        if x > my_max:
            my_max = x
    return my_max

def merge_sort(arr):
    if len(arr) <= 1:
        return list(arr)

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def ft_percentile(sorted_v, p):
    n = ft_count(sorted_v)

    if n == 0:
        return None
    
    pos = (n - 1) * p
    low = int(pos)
    high = min(low + 1, n - 1)

    return sorted_v[low] + (sorted_v[high] - sorted_v[low]) * (pos - low)

def ft_skew(v):
    m = ft_mean(v)
    sd = math.sqrt(ft_sum([(x - m) ** 2 for x in v]) / ft_count(v))
    return 0.0 if sd == 0 else ft_sum([((x - m) / sd) ** 3 for x in v]) / ft_count(v)


def ft_quartiles(sorted_v):
    """Return (q1, q2, q3) corresponding to 25%, 50%, and 75%."""
    q1 = ft_percentile(sorted_v, 0.25)
    q2 = ft_percentile(sorted_v, 0.50)
    q3 = ft_percentile(sorted_v, 0.75)
    return q1, q2, q3


def compute(v, bonus):
    s = merge_sort(v)
    q1, q2, q3 = ft_quartiles(s)

    stats = {
        "Count": ft_count(v),
        "Mean": ft_mean(v),
        "Std": ft_std(v),
        "Min": s[0],
        "25%": q1,
        "50%": q2,
        "75%": q3,
        "Max": s[-1],
    }

    if bonus:
        stats.update({
            "Var": ft_var(v),
            "Range": s[-1] - s[0],
            "IQR": q3 - q1,
            "Skew": ft_skew(v),
        })

    return stats

SHORT = {
    "Arithmancy": "Arithm",
    "Astronomy": "Astron",
    "Herbology": "Herbol",
    "Defense Against the Dark Arts": "Defense",
    "Divination": "Divin",
    "Muggle Studies": "Muggle",
    "Ancient Runes": "Runes",
    "History of Magic": "History",
    "Transfiguration": "Transf",
    "Potions": "Potions",
    "Care of Magical Creatures": "Creatur",
    "Charms": "Charms",
    "Flying": "Flying"
}


def ft_format(val):
    if val is None:
        return "N/A"
    if isinstance(val, int) and not isinstance(val, bool):
        return str(val)
    return f"{val:.2f}"


def print_table(names, stats):
    if not names:
        return

    labels = list(stats[names[0]])
    label_w = max(len("Feature"), *(len(lbl) for lbl in labels)) + 1

    headers = [SHORT.get(n, n[:8]) for n in names]
    cols = [[ft_format(stats[n][lbl]) for lbl in labels] for n in names]
    widths = [
        max(len(h), *(len(cell) for cell in col)) + 2
        for h, col in zip(headers, cols)
    ]

    print("Feature".ljust(label_w) + "".join(h.rjust(w) for h, w in zip(headers, widths)))
    for i, lbl in enumerate(labels):
        line = lbl.ljust(label_w) + "".join(col[i].rjust(w) for col, w in zip(cols, widths))
        print(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("-b", "--bonus", action="store_true")
    args = parser.parse_args()

    try:
        df = load(args.filename)
    except FileNotFoundError:
        print(f"Error: file not found: {args.filename}", file=sys.stderr)
        return
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        return

    names = [col for col in df.select_dtypes(include=[np.number]).columns if col != "Index"]
    if not names:
        print("Error: no numerical features found", file=sys.stderr)
        return

    total = len(df)
    stats = {}
    for name in names:
        values = [float(v) for v in df[name] if not pd.isna(v)]
        if len(values) < 2:
            continue
        stats[name] = compute(values, args.bonus)
        if args.bonus:
            stats[name]["Missing"] = total - stats[name]["Count"]

    if not stats:
        sys.exit("Error: no numerical features found")

    print_table(list(stats), stats)


if __name__ == "__main__":
    main()






