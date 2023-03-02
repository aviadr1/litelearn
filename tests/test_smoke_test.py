import pickle

import pytest
import seaborn as sns
import litelearn as ll
import pandas as pd
import numpy as np
import pickle


def test_basic_usage():
    dataset = "penguins"
    target = "body_mass_g"

    df = sns.load_dataset(dataset)
    df = df.dropna(subset=[target])
    model = ll.core_regress_df(df, target)
    result = model.get_evaluation()
    model.display_evaluations()
    pred = model.predict(df)


def test_train_index():
    dataset = "penguins"
    target = "species"

    df = sns.load_dataset(dataset)
    df = df.dropna(subset=[target])
    train = df.head(250)
    test = df.drop(train.index)
    model, pool = ll.core_classify_df(df, target, train_index=train.index)
    # result = model.get_evaluation()
    # model.display_evaluations()
    assert (model.X_train.index == train.index).all()
    assert (model.y_train.index == train.index).all()
    assert (model.X_test.index == test.index).all()
    assert (model.y_test.index == test.index).all()


@pytest.fixture
def dowjones_dataset():
    dataset = "dowjones"

    df = sns.load_dataset(dataset)
    df = df.dropna(subset=["Price"])

    # assert df["Date"].dtype == "date"
    df["Date"] = pd.to_datetime(df["Date"])
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    # calculate moving average
    df["prev_5d_avg"] = df.rolling(window=5).mean()
    return df


def test_dates_regression(dowjones_dataset):
    target = "Price"
    df = dowjones_dataset
    model = ll.core_regress_df(df, target, iterations=20, use_nulls=True)
    result = model.get_evaluation()
    model.display_evaluations()
    y_pred = model.predict(df)


def test_dates_classification(dowjones_dataset):
    target = "up"
    df = dowjones_dataset
    df["up"] = df["Price"] > df["prev_5d_avg"]
    df = df.dropna(subset=[target])
    model, pool = ll.core_classify_df(df, target)
    # result = model.get_evaluation()
    # model.display_evaluations()


def test_native_nulls_regression():
    dataset = "penguins"
    target = "body_mass_g"

    df = sns.load_dataset(dataset)
    df = df.dropna(subset=[target])
    train = df.head(250)
    test = df.drop(train.index)
    model = ll.core_regress_df(df, target, train_index=train.index, use_nulls=False)


def test_native_nulls_classification():
    dataset = "penguins"
    target = "species"

    df = sns.load_dataset(dataset)
    df = df.dropna(subset=[target])
    train = df.head(250)
    test = df.drop(train.index)
    model, pool = ll.core_classify_df(
        df, target, train_index=train.index, use_nulls=False
    )


def test_catboost_nulls_classification():
    dataset = "penguins"
    target = "species"

    df = sns.load_dataset(dataset)
    df = df.dropna(subset=[target])
    train = df.head(250)
    test = df.drop(train.index)
    model, pool = ll.core_classify_df(
        df, target, train_index=train.index, use_nulls=True
    )
    y_pred = model.predict(df)


def test_categories():
    df = pd.DataFrame(
        {
            "cat": ["a", "b", "c", "d", "e", "f"],
            "int": [1, 2, 3, 4, 5, 6],
            "int_with_nulls": [1, np.nan, 3, np.nan, 2, 1],
            "floats": [np.nan, 1.2, 1.3, 1.4, 1.3, 1.2],
            "bools": [False, False, False, True, True, True],
            "value": [1, 2, 2, 1, 1, 2],
        }
    )
    target = "value"
    model, pool = ll.core_classify_df(df, target, use_nulls=True)
    y_pred = model.predict(df)
    train_index = model.train_frame.X_train.index
    assert y_pred[train_index].tolist() == df.value[train_index].tolist()


def test_pickle_regression():
    dataset = "penguins"
    target = "body_mass_g"

    df = sns.load_dataset(dataset)
    df = df.dropna(subset=[target])
    model = ll.core_regress_df(df, target, use_nulls=True)
    storage = pickle.dumps(model)
    model2 = pickle.loads(storage)
    model2.predict(df)


def test_pickle_classification():
    dataset = "penguins"
    target = "island"

    df = sns.load_dataset(dataset)
    df = df.dropna(subset=[target])
    model, pool = ll.core_classify_df(df, target, use_nulls=True)
    storage = pickle.dumps(model)
    model2 = pickle.loads(storage)
    result = model2.predict(df)
    assert isinstance(result, pd.Series)


@pytest.mark.xfail
def test_predict_with_custom_nulls():
    dataset = "penguins"
    target = "body_mass_g"

    df = sns.load_dataset(dataset)
    df = df.dropna(subset=[target])
    model = ll.core_regress_df(
        df,
        target,
        use_nulls=False,  # litelearn will fillna and create _is_missing fields
    )
    model.predict(df)  # TODO: handle missing values properly


def test_predict_proba():
    dataset = "penguins"
    target = "species"

    df = sns.load_dataset(dataset)
    df = df.dropna(subset=[target])
    model, pool = ll.core_classify_df(
        df,
        target,
        use_nulls=True,
    )
    proba = model.predict_proba(df)
    assert proba.shape == (len(df), 3)
    assert set(proba.columns) == set(df.species.unique())
    assert proba.index.equals(df.index)
