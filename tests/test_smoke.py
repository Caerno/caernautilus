'''Smoke tests: every public helper has to survive the current numpy/pandas/sklearn.'''

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from caernautilus import classes as C
from caernautilus import output as O


@pytest.fixture
def df():
    '''Titanic-shaped frame: numeric with NA, low- and high-cardinality objects,
       and a target where the NA rows stand for the test part.'''
    rng = np.random.default_rng(0)
    n = 200
    data = pd.DataFrame({
        "age": rng.normal(30, 10, n).round(1),
        "fare": rng.gamma(2, 20, n).round(2),
        "pclass": rng.integers(1, 4, n),
        "sex": rng.choice(["male", "female"], n),
        "deck": rng.choice(list("ABCDEFG"), n),
        "ticket": [f"T{i:04d}" for i in range(n)],
        "survived": rng.integers(0, 2, n).astype(float),
    })
    data.loc[data.sample(30, random_state=1).index, "age"] = np.nan
    data.loc[data.sample(60, random_state=3).index, "fare"] = np.nan
    data.loc[data.sample(40, random_state=2).index, "survived"] = np.nan
    return data


@pytest.fixture
def X(df):
    return df.drop(columns="survived")


def test_informer(df):
    info = O.informer(df, "survived")
    assert (info["train_obs"], info["test_obs"], info["n_cls"]) == (160, 40, 2)
    O.informer_print(info)


def test_imperfection(df):
    full = O.imperfection(df, numeric=True)
    assert set(full.index) == set(df.columns)
    assert full.loc["age", "NA Share"] == pytest.approx(0.15)
    assert full.loc["sex", "Num. of unique"] == 2
    # default view keeps non-numeric columns only, no_nan keeps the gappy ones only
    assert "age" not in O.imperfection(df).index
    assert set(O.imperfection(df, numeric=True, no_nan=True).index) == {"age", "fare", "survived"}


def test_multicolumn(df):
    assert O.multicolumn(df.sample(1, random_state=0).T, cols=3) is not None


def test_plot_conf_map():
    hm, precision, recall = O.plot_conf_map(np.array([[50, 5], [7, 38]]), title="t")
    assert precision == pytest.approx([50 / 57, 38 / 43])
    assert recall == pytest.approx([50 / 55, 38 / 45])
    assert 0 < hm < 1


def test_plot_conf_map_empty_class():
    '''A class nothing was predicted for must not blow up on division.'''
    hm, precision, _ = O.plot_conf_map(np.array([[55, 0], [7, 0]]), blind=True)
    assert precision[1] == 0
    assert hm == pytest.approx(0)


def test_plot_some_scatters(df):
    O.plot_some_scatters(df[["age", "fare", "pclass"]].fillna(0).values,
                         df["survived"].fillna(0).values)


def test_img_roundtrip():
    rng = np.random.default_rng(0)
    img = rng.random((32, 32, 4))
    assert np.allclose(O.img_set_up(O.img_breakdown(img)), img)
    assert np.allclose(O.img_set_left(O.img_breakright(img)), img)
    squeezed = O.img_squeeze(img, PCA, 4, (O.img_breakdown, O.img_set_up))
    assert squeezed.shape == img.shape and squeezed.min() >= 0 and squeezed.max() <= 1
    O.img_framaker(img, 4, PCA)


def test_nanfixer(X):
    fixed = C.NanFixer().fit_transform(X)
    assert not fixed.isna().any().any()
    assert list(fixed.columns) == list(X.columns)
    # fare is over the 0.225 limit -> fuzzy fill, age under it -> plain mean
    assert X["fare"].isna().sum() > 0
    assert fixed.loc[X["age"].isna(), "age"].nunique() == 1
    assert fixed.loc[X["fare"].isna(), "fare"].nunique() > 1


def test_nanfixer_passes_unplanned_dtypes_through():
    X = pd.DataFrame({"flag": [True, False, True], "when": pd.to_datetime(["2022-01-01"] * 3)})
    assert C.NanFixer().fit_transform(X)["flag"].tolist() == [True, False, True]


def test_not_fitted_message():
    with pytest.raises(NotFittedError, match="Digitalize"):
        C.Digitalize().transform(pd.DataFrame({"a": [1]}))


def test_digitalize(X):
    out = C.Digitalize().fit_transform(X.fillna(0))
    assert len(out) == len(X)
    assert out.select_dtypes("object").empty
    assert "sex_male" in out.columns          # 2 values -> one-hot
    assert "ticket_top5" in out.columns       # 200 values -> top-n rating
    assert out["ticket_top5"].max() == 5


@pytest.mark.parametrize("mode", ["raw", "cut", "qcut"])
def test_digitalize_modes(X, mode):
    out = C.Digitalize(mode=mode, ncut=4).fit_transform(X.fillna(0))
    assert out["age"].nunique() <= (4 if mode != "raw" else len(X))


def test_pipeline_and_clone(X):
    pipe = Pipeline([("nan", C.NanFixer()), ("dig", C.Digitalize())])
    out = clone(pipe).fit_transform(X)
    assert not out.isna().any().any()


def test_featuretrans_numeric(df):
    names = [s.name for s in C.FeatureTrans(df["fare"].fillna(1.0))]
    assert "fare_sqrt" in names and "fare_qcut5" in names and "fare_top5" in names


def test_featuretrans_categorical(df):
    generated = list(C.FeatureTrans(df["deck"]))
    counted = next(s for s in generated if s.name == "deck_count")
    assert (counted == df["deck"].map(df["deck"].value_counts())).all()
    shares = next(s for s in generated if s.name.startswith("deck_freq"))
    assert 0 < shares.min() and shares.max() <= 1


def test_featuretrans_bilabel():
    out = C.FeatureTrans.bilabel(pd.Series(["Yes", "No", "Yes"], name="a"))
    assert out.tolist() == [1, 0, 1]
    assert C.FeatureTrans.bilabel(pd.Series(["l", "r"], name="a")).tolist() == [0, 1]


@pytest.fixture
def line():
    '''y = 3x + 5 with a bit of noise.'''
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.uniform(1, 10, 300)})
    return X, 3 * X["x"] + 5 + rng.normal(0, 0.5, 300)


def test_slow_poly_linear_reg():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.uniform(1, 10, 300)})
    y = 3 * X["x"] ** 2 - 2 * X["x"] + 5 + rng.normal(0, 1, 300)
    model = C.SlowPolyLinearReg(gener=[0, 1, 2], norm="std", alpha=0.01).fit(X, y)
    assert model.score(X, y) > 0.99
    assert len(model.predict(X)) == len(X)


@pytest.mark.parametrize("norm", [None, "minmax", "std", "max", "mean"])
def test_intercept_survives_a_shuffled_index(line, norm):
    '''The constant column used to carry its own RangeIndex, so anything that
       reindexes X - train_test_split above all - broke the fit.'''
    X, y = line
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)
    model = C.SlowPolyLinearReg(gener=[0, 1], norm=norm).fit(X_tr, y_tr)
    assert model.score(X_te, y_te) > 0.99
    assert model.predict(X_te).index.equals(X_te.index)


def test_intercept_is_recovered(line):
    X, y = line
    model = C.SlowPolyLinearReg(gener=[0, 1]).fit(X, y)
    assert np.ravel(model.w) == pytest.approx([5, 3], abs=0.1)


def test_default_fits_an_intercept(line):
    '''The old default was x alone, which forced the line through the origin.'''
    X, y = line
    assert np.ravel(C.SlowPolyLinearReg().fit(X, y).w) == pytest.approx([5, 3], abs=0.1)


def test_score_is_r2_not_squared_correlation(line):
    '''corr**2 ignores bias and scale: a model off by a constant still scored ~1.'''
    X, y = line
    model = C.SlowPolyLinearReg(gener=[0, 1]).fit(X, y)
    assert model.score(X, y) == pytest.approx(r2_score(y, model.predict(X)))
    model.w = np.asarray(model.w) * 2 + 100
    assert model.score(X, y) == pytest.approx(r2_score(y, model.predict(X)))
    assert model.score(X, y) < 0


def test_refit_resets_normalization(line):
    X, y = line
    model = C.SlowPolyLinearReg(gener=[0, 1], norm="minmax").fit(X, y)
    model.fit(X * 100, y)
    assert float(np.ravel(model.norm_max)[-1]) > 100


def test_na_is_refused(line):
    X, y = line
    X = X.copy()
    X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NA values"):
        C.SlowPolyLinearReg(gener=[0, 1]).fit(X, y)


def test_singular_input_still_fits():
    '''inv() used to raise and leave the object without weights at all.'''
    X = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 4, 6, 8]})   # b == 2a
    model = C.SlowPolyLinearReg(gener=[1]).fit(X, pd.Series([1.0, 2, 3, 4]))
    assert len(model.predict(X)) == 4


def test_slow_poly_linear_reg_rejects_bad_generator():
    model = C.SlowPolyLinearReg(gener=["log2"])
    with pytest.raises(ValueError):
        model.fit(pd.DataFrame({"x": [-1.0, 2.0]}), pd.Series([1.0, 2.0]))
