# caernautilus

Caerno utilities — a small toolbox of things I kept re-typing in every EDA notebook:
quick dataset reports, a few plots, and sklearn-compatible encoders that can be
dropped straight into a `Pipeline` / `GridSearchCV`.

Nothing here is a framework. Each helper is one function or one class you can read in a minute.

## Install

```bash
pip install git+https://github.com/Caerno/caernautilus.git
```

On Kaggle / Colab the same line works in a cell (`!pip install git+...`).
For a kernel with no internet, add
[the archive GitHub builds for every commit](https://github.com/Caerno/caernautilus/archive/refs/heads/main.zip)
as a dataset and install it from disk:

```bash
pip install --no-deps /kaggle/input/<your-dataset>/caernautilus-main.zip
```

Requires Python ≥ 3.9, numpy, pandas, scikit-learn, matplotlib, seaborn.

## Quick start

```python
import pandas as pd
from caernautilus import informer, informer_print, imperfection, NanFixer, Digitalize

df = pd.concat([train, test])          # test rows have no target -> NaN

informer_print(informer(df, "Survived"))
#   Features:       12
#   Observations:   891/418
#   train dataset:  68.0%
#   classes:        0: 61.6%, 1: 38.4%

imperfection(df)                       # what is missing and how varied it is
```

`imperfection` returns a frame sorted by NA share — share of missing values, number of
unique values, dtype and the values themselves. By default it shows non-numeric columns
only; `numeric=True` adds the rest, `no_nan=True` keeps only columns that actually have gaps.

## Encoders

Both are `BaseEstimator` + `TransformerMixin`, so `fit`/`transform`/`fit_transform`,
`get_params`, `clone` and grid search over their parameters all work.

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ("nan", NanFixer()),               # fill the gaps
    ("dig", Digitalize()),             # make everything numeric
    ("clf", RandomForestClassifier()),
])

grid = GridSearchCV(pipe, {
    "nan__lim":    [0.1, 0.225, 0.5],
    "nan__method": ["mean", "median"],
    "dig__mode":   ["raw", "cut", "qcut"],
    "dig__ncut":   [5, 10],
})
```

**`NanFixer(lim=0.225, method="mean", filler="U")`** — picks a filling strategy per column
at `fit` time and applies it at `transform` time:

| column | condition | strategy |
|---|---|---|
| numeric | NA share ≤ `lim` | `method` (`mean`/`median`) |
| numeric | NA share > `lim` | random draw from the column's own normal distribution, clipped to its range |
| object | one value dominates (>25%) | mode |
| object | otherwise | the `filler` string |

Columns of any other dtype pass through untouched.

**`Digitalize(alim=10, mode="raw", ncut=5)`** — turns a mixed frame into a numeric one:

| column | condition | result |
|---|---|---|
| numeric | < 3 unique values | as is |
| any | < `alim` unique values | one-hot (`pd.get_dummies`) |
| numeric | `mode="cut"` / `"qcut"` | binned into `ncut` equal-spaced / equal-sized bins |
| numeric | `mode="raw"` | as is |
| non-numeric | ≥ `alim` unique values | top-`ncut` frequency rating (most frequent → 1, the tail → `ncut`) |

**`FeatureTrans(series)`** is the shared kit both encoders are built from, and it is
iterable: it yields every transformation that makes sense for that column's dtype —
handy when you want to throw candidate features at a selector.

```python
candidates = pd.concat(list(FeatureTrans(df["Fare"])), axis=1)
# Fare_count, Fare_freqn, Fare_top5..., Fare_cut5..., Fare_qcut5..., Fare_sqrt, Fare_log2, Fare_pow2 ...
```

Its pieces are static methods you can also call directly: `count`, `freq`, `top`,
`cut`, `qcut`, `math`, `bilabel`, `fuzzy_mean`, `robust`, `fill`.

## Plots

```python
from caernautilus import plot_conf_map, plot_some_scatters

hm, precision, recall = plot_conf_map(confusion_matrix(y_true, y_pred), title="RF")
```

`plot_conf_map` draws the confusion matrix as shares, with a precision row, a recall
column and — in the corner — the harmonic mean of all of them. That last number is the
point of it: a class-agnostic generalization of F1 for when no class is "the positive one".
Returns `(harmonic_mean, precision, recall)`; `blind=True` computes without drawing,
which is what you want inside a scoring loop.

`plot_some_scatters(X, y)` — pairwise scatter plots of every column combination, coloured by `y`.

`multicolumn(series, cols=5)` folds a long series into several columns so a wide
`df.sample().T` fits on one screen.

## Image compression demo

`img_squeeze` reduces the dimensionality of an RGBA image by unrolling channels into a
flat 2D matrix, running any sklearn decomposition on it and rebuilding the picture:

```python
from sklearn.decomposition import PCA
from caernautilus import img_framaker

img_framaker(img, 20, PCA)   # image restored from 20 components
```

## Estimator from scratch

`SlowPolyLinearReg(gener=[1], norm=None, alpha=None)` — linear regression written on
numpy/pandas, with polynomial and non-linear feature generation (`0` = intercept,
integers = powers, `"sqrt"`/`"cbrt"`/`"log2"`/`"ln"`/`"log10"`), optional normalization
(`"minmax"`, `"std"`, `"max"`, `"mean"`) and L2 regularization. Closed-form solution,
so it warns you about singular and ill-conditioned matrices instead of hiding them.
It exists to show the mechanics, not to compete with `sklearn.linear_model`.

```python
model = SlowPolyLinearReg(gener=[0, 1, 2], norm="std", alpha=0.01)
model.fit(X, y)
model.score(X, y)   # R²
```

## Input

`number("How many? ", t=int, op=">=", limit=0)` — keeps asking until the answer parses
and satisfies the condition.

## Development

```bash
pip install -e .[test]
pytest
```

The drop-in zip in `download/` is a plain stdlib call, no notebook needed:

```bash
python -m zipfile -c download/caernautilus.zip __init__.py classes.py input.py output.py
```
