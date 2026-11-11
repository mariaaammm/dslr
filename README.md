# DSLR: Harry Potter and the Data Scientist

A project of the **42 school** curriculum (*Data Science × Logistic Regression*).

The Sorting Hat has been bewitched, so we rebuild it: a **one-vs-rest logistic regression**, trained from scratch with **gradient descent**, that predicts a student's Hogwarts house (Gryffindor, Hufflepuff, Ravenclaw, Slytherin) from their course scores. The repository also contains the data-exploration programs (describe, histogram, scatter plot, pair plot) used to understand the data before training.

## Contents

| File | Purpose |
|---|---|
| `describe.py` | Statistics of every numerical feature (count, mean, std, min, 25%, 50%, 75%, max), all computed by hand |
| `histogram.py` | Score histograms per course and house |
| `scatter_plot.py` | Scatter plot of the two most similar features |
| `pair_plot.py` | Pair plot of all courses, coloured by house |
| `logreg_train.py` | Trains the model and writes `weights.json` |
| `logreg_predict.py` | Reads `weights.json` and writes `houses.csv` |
| `utils.py` | Shared constants (`HOUSES`, `COURSES`, `FEATURES`, `COLORS`), `load`, `StandardScaler`, `sigmoid` |

## Requirements

Python 3.10+ and:

```
pip install -r requirements.txt     # numpy, pandas, matplotlib, seaborn, tabulate
```

`tabulate` is used only to format the table printed by `describe.py`.

## Usage

```
python describe.py dataset_train.csv          # add -b / --bonus for Var, Range, IQR, Skew, Missing
python histogram.py dataset_train.csv
python scatter_plot.py dataset_train.csv
python pair_plot.py dataset_train.csv
python logreg_train.py dataset_train.csv      # writes weights.json
python logreg_predict.py dataset_test.csv weights.json   # writes houses.csv
```

`houses.csv` has the required format:

```
Index,Hogwarts House
0,Hufflepuff
1,Ravenclaw
...
```

## Data

`dataset_train.csv` has 1600 students, `dataset_test.csv` has 400 (the `Hogwarts House` column is empty there). Each row has a name, birthday, best hand and 13 course scores. Some scores are missing. Only the 13 course scores are numerical features; names, birthday and best hand are not used.

## What the analysis shows

- **`describe.py`:** The courses have very different scales (Arithmancy is in the tens of thousands, Herbology and Care of Magical Creatures are in single digits), so the features must be standardized before gradient descent. Every statistic is computed by hand (`reduce`, list comprehensions, own merge sort and percentile interpolation); the values match `pandas.describe()` to the 6th decimal.
- **Histogram:** *Arithmancy* has the most homogeneous distribution between the four houses (followed by *Care of Magical Creatures*): the four histograms overlap almost completely, so these courses say little about the house.
- **Scatter plot:** *Astronomy* and *Defense Against the Dark Arts* are the two similar features (Pearson r = -1.0000); one of them is redundant.
- **Pair plot:** most other courses show clearly separated clusters for the houses, so they are useful for classification.

## Model

- **One-vs-rest:** one binary logistic regression per house ("this house" vs "the others"). A student is assigned to the house whose model returns the highest probability.
- **Hypothesis and loss:** `h = sigmoid(X·w + b)`, cross-entropy loss. Probabilities are clipped to avoid `log(0)`, and the input of the sigmoid is clipped to avoid overflow.
- **Gradient descent (batch):** `w -= lr * Xᵀ(h - y) / m`, `b -= lr * sum(h - y) / m`. Defaults: learning rate 0.1, 1000 iterations, weights initialised to zero.
- **Features:** all 13 course scores are given to the model. The analysis above suggests that Arithmancy and Care of Magical Creatures carry almost no information and that Astronomy duplicates Defense Against the Dark Arts; training on the 9 courses listed in `FEATURES` (`utils.py`) gives the same accuracy.
- **Missing values:** the scaler ignores missing scores when computing the mean and standard deviation of each feature (`nanmean`, `nanstd`), and replaces a missing score by the training mean (a z-score of 0). Rows are never dropped, because every student of the test set must get a house.
- **Standardization:** every feature is standardized (z-score) with the training mean and standard deviation. They are saved in `weights.json`, so `logreg_predict.py` applies exactly the same transformation.
- **`weights.json`:** the feature list, the scaler's mean and std, and the weights and bias of the four models.

## Results

On a random 80 % / 20 % split of `dataset_train.csv` (scaler and models fitted on the 80 % only) the accuracy on the unseen 20 % is about **98.4 - 98.8 %**, depending on the split, above the 98 % required by the subject.

## Notes

- Everything is implemented from scratch with NumPy; no machine-learning library is used for training or prediction.