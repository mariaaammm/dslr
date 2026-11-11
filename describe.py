from functools import reduce
import math
import csv
import argparse
from tabulate import tabulate 


def ft_count(v): return reduce(lambda n, _: n + 1, v, 0)
def ft_sum(v):   return reduce(lambda s, x: s + x, v, 0.0)
def ft_mean(v):  return ft_sum(v) / ft_count(v)

def ft_var(v):
    m = ft_mean(v)
    return ft_sum([(x - m) ** 2 for x in v]) / (ft_count(v) - 1)

def ft_std(v): return math.sqrt(ft_var(v))
def ft_min(v): return reduce(lambda a, b: b if b < a else a, v)
def ft_max(v): return reduce(lambda a, b: b if b > a else a, v)


def ft_sort(v):
    """Merge sort (no sorted()/sort())."""
    if len(v) <= 1:
        return list(v)
    mid = len(v) // 2
    left, right = ft_sort(v[:mid]), ft_sort(v[mid:])
    out, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    return out + left[i:] + right[j:]


def ft_percentile(sorted_v, p):
    """Linear interpolation between closest ranks."""
    k = (ft_count(sorted_v) - 1) * p
    lo = int(k)
    hi = min(lo + 1, ft_count(sorted_v) - 1)
    return sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * (k - lo)

def ft_skew(v):
    m = ft_mean(v)
    sd = math.sqrt(ft_sum([(x - m) ** 2 for x in v]) / ft_count(v))
    return 0.0 if sd == 0 else ft_sum([((x - m) / sd) ** 3 for x in v]) / ft_count(v)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def read_numeric_features(path):
    """Return ({feature: [floats without empty cells]}, total_rows)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    values = {}
    for name in rows[0]:
        if name == "Index":
            continue
        cells = [row[name] for row in rows if row[name].strip() != ""]
        if cells and all(is_number(c) for c in cells):
            values[name] = [float(c) for c in cells]

    return values, len(rows)


def compute(v, bonus):
    s = ft_sort(v)
    q1, q2, q3 = ft_percentile(s, 0.25), ft_percentile(s, 0.50), ft_percentile(s, 0.75)
    stats = {"Count": ft_count(v), "Mean": ft_mean(v), "Std": ft_std(v), "Min": s[0],
             "25%": q1, "50%": q2, "75%": q3, "Max": s[-1]}
    if bonus:
        stats.update({"Var": ft_var(v), "Range": s[-1] - s[0], "IQR": q3 - q1, "Skew": ft_skew(v)})
    return stats


def main():
    parser = argparse.ArgumentParser(description="Describe the numerical features of a dataset")
    parser.add_argument("filename")
    parser.add_argument("-b", "--bonus", action="store_true",
                        help="show extra rows (Var, Range, IQR, Skew, Missing)")
    args = parser.parse_args()

    try:
        data, total = read_numeric_features(args.filename)
    except FileNotFoundError:
        print(f"Error: file not found: {args.filename}")
        return
    if not data:
        print("Error: no numerical features found")
        return

    names = list(data)
    stats = {n: compute(data[n], args.bonus) for n in names}
    if args.bonus:
        for n in names:
            stats[n]["Missing"] = total - stats[n]["Count"]

    # labels = list(stats[names[0]])
    # widths = [max(len(n), 10) + 1 for n in names]
    # print(" " * 8 + "".join(n[: w - 1].rjust(w) for n, w in zip(names, widths)))

    labels = list(stats[names[0]])
    rows = [[label] + [stats[n][label] for n in names] for label in labels]
    print(tabulate(rows, headers=[""] + names, floatfmt=".2f"))

    # for label in labels:
    #     print(label.ljust(8) + "".join(f"{stats[n][label]:.6f}".rjust(w) for n, w in zip(names, widths)))


if __name__ == "__main__":
    main()