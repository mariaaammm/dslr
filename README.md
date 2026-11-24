# dslr — Data Science × Logistic Regression, a 42 School project

*Harry Potter and the Data Scientist.* This is a project of the [42 School](https://42.fr) curriculum.

The Sorting Hat is broken, so we rebuild it: a **one-vs-all logistic regression**, written from scratch with
gradient descent, that predicts the Hogwarts house of a student from the scores of their courses.

The project has two parts: a data analysis (`describe`, `histogram`, `scatter_plot`, `pair_plot`)
and the classifier itself (`logreg_train`, `logreg_predict`).

| | |
|---|---|
| Result | **0.990** on the official test set (396 of 400 students, required: 0.98) |
| Model | one logistic regression per house, the highest score wins |
| Libraries | `numpy`, `pandas`, `matplotlib`, `seaborn` |

## Contents

| File | Purpose |
|---|---|
| `describe.py` | Statistics of every numerical column (count, mean, std, min, quartiles, max), written by hand |
| `histogram.py` | Score distribution of every course per house and the most homogeneous course |
| `scatter_plot.py` | The two most similar features (largest absolute Pearson correlation) |
| `pair_plot.py` | Scatter plot matrix of all courses, to choose the features |
| `logreg_train.py` | Trains the model and writes `weights.json` |
| `logreg_predict.py` | Reads `weights.json`, predicts the houses of a dataset and writes `houses.csv` |
| `utils.py` | Shared code: data loading and checks, constants, `StandardScaler`, `sigmoid` |
| `evaluate.py`, `dataset_truth.csv` | The official checking script of the subject and the true answers of the test set |
| `notebooks/` | Exploratory data analysis |
| `datasets/` | `dataset_train.csv` (1600 students) and `dataset_test.csv` (400 students) |

## Installation

Python 3.10 or newer is needed.

```bash
git clone https://github.com/mariaaammm/dslr.git
cd dslr
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Data analysis

```bash
python describe.py datasets/dataset_train.csv
python histogram.py datasets/dataset_train.csv
python scatter_plot.py datasets/dataset_train.csv
python pair_plot.py datasets/dataset_train.csv
```

`histogram.py` prints the most homogeneous course, `scatter_plot.py` prints the most similar pair of features:

```
Most homogeneous course: Arithmancy (0.068)
Most similar features: Astronomy / Defense Against the Dark Arts (r = -1.0000)
```

### 2. Training

```bash
python logreg_train.py datasets/dataset_train.csv
```

This writes `weights.json`: the selected features, the standardization (mean and standard deviation of every
feature) and the weights and bias of the four classifiers.

### 3. Prediction

```bash
python logreg_predict.py datasets/dataset_test.csv weights.json
```

This writes `houses.csv`:

```
Index,Hogwarts House
0,Hufflepuff
1,Ravenclaw
...
```

### 4. Check

```bash
python evaluate.py
# Your score on test set: 0.990
```

### Bonus

```bash
# more fields in describe: Var, Range, IQR, Skew, Missing
python describe.py datasets/dataset_train.csv -b

# other optimization algorithms (default: batch gradient descent)
python logreg_train.py datasets/dataset_train.csv --method sgd
python logreg_train.py datasets/dataset_train.csv --method minibatch
python logreg_train.py datasets/dataset_train.csv --method minibatch --batch-size 64 --epochs 300

# another weights file
python logreg_train.py datasets/dataset_train.csv -o my_weights.json
```

| `--method` | Update | Default epochs |
|---|---|---|
| `batch` | the whole dataset per update | 1000 |
| `sgd` | one student per update | 30 |
| `minibatch` | `--batch-size` students per update (default 32) | 200 |

All three methods reach the same score on the test set (0.990).

## Data analysis

The dataset has 13 courses. The analysis decides which of them are worth giving to the model.

**Histogram.** A course is *homogeneous* if the four houses have the same score distribution there: it cannot tell
the houses apart. To measure it, `histogram.py` computes
`(largest house mean − smallest house mean) / standard deviation of the course`. The smaller the number, the more homogeneous the course.

| Course | Homogeneity | Used by the model |
|---|---|---|
| Arithmancy | 0.068 | no |
| Care of Magical Creatures | 0.147 | no |
| Ancient Runes | 1.859 | yes |
| Herbology | 1.879 | yes |
| Defense Against the Dark Arts | 1.908 | yes |
| Astronomy | 1.911 | no (duplicate, see below) |
| Muggle Studies | 2.037 | yes |
| Potions | 2.075 | yes |
| History of Magic | 2.220 | yes |
| Transfiguration | 2.284 | yes |
| Divination | 2.367 | yes |
| Charms | 2.465 | yes |
| Flying | 2.656 | yes |

Arithmancy and Care of Magical Creatures are about 15 to 40 times more homogeneous than the others.

**Scatter plot.** *Astronomy* and *Defense Against the Dark Arts* have a correlation of **r = −1.0000**: the points lie
on a straight line (`Astronomy ≈ −100 × Defense Against the Dark Arts`). The two courses carry the same information,
so one of them is enough.

**Pair plot.** In the scatter plot matrix the houses form separate clusters for the remaining courses, which
is why a linear classifier works well here. Arithmancy and Care of Magical Creatures stay mixed in every panel.

**Selected features (10):**
Herbology, Defense Against the Dark Arts, Divination, Muggle Studies, Ancient Runes,
History of Magic, Transfiguration, Potions, Charms, Flying.

## The model

- **One-vs-all.** For each house a logistic regression answers "is this student in this house?". For a new student
  the four probabilities are computed and the house with the highest one is chosen.
- **Hypothesis:** `h = sigmoid(X·w + b)`.
- **Loss:** cross-entropy, minimized by gradient descent (learning rate 0.1).
- **Standardization.** Every feature is transformed to `(x − mean) / std`, with the mean and the standard deviation
  of the **training** set. They are saved in `weights.json`, so that `logreg_predict.py` applies exactly the same
  transformation.
- **Missing scores.** A missing score (NaN) is replaced by the mean of its feature, which is the value 0 after
  standardization. The same rule is used in training and in prediction.
- **Random order.** In `sgd` and `minibatch` the students are shuffled every epoch with a fixed seed (42),
  so the result is reproducible.

### Results

| Measure | Accuracy |
|---|---|
| Official test set (`evaluate.py`) | **0.990** (396 / 400) |
| 5-fold cross-validation, training folds | 98.19 % |
| 5-fold cross-validation, validation folds | 98.19 % |

The accuracy on the validation folds is the same as on the training folds and it does not change when the number
of epochs grows from 50 to 5000. The model does not overfit. The students that are wrongly classified
are the same ones in every run, so the remaining error is a limit of the data for a linear model.

## Error handling

The programs never show a traceback for a wrong input. They print a short message and stop:

- a missing, empty or unreadable file or a file without the required columns;
- a corrupted `weights.json`;
- `logreg_train.py` on a file without house labels (for example `dataset_test.csv`);
- values that must be positive (`--epochs`, `--batch-size`).

## Notes

- `describe.py` does not use `pandas.describe()`, `mean()`, `std()`, `min()`, `max()`, `percentile()` or similar
  functions: count, mean, standard deviation, minimum, maximum and the percentiles
  (with linear interpolation, like pandas) are computed by hand. The result was compared with `pandas.describe()`
  and it is the same up to floating point precision.
- `weights.json` and `houses.csv` are generated by the programs and can be created again at any time.