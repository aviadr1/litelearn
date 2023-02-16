import pytest
import seaborn as sns
import litelearn as ll
import pandas as pd


def test_basic_usage():
    dataset = "penguins"
    target = "body_mass_g"

    df = sns.load_dataset(dataset)
    df = df.dropna(subset=[target])
    model = ll.core_regress_df(df, target)
    result = model.get_evaluation()
    model.display_evaluations()


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
    model = ll.core_regress_df(df, target)
    result = model.get_evaluation()
    model.display_evaluations()


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
