# %pip install --quiet --upgrade catboost googledrivedownloader shap altair

import logging
from typing import *

# For transformations and predictions
import pandas as pd
import shap


# For validation
from sklearn.model_selection import train_test_split as split

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig()
shap.initjs()


def print_version(lib):
    print(lib.__name__, lib.__version__)


def default_value(value, default, marker=None):
    return (value, default)[value == marker]


def _is_interactive():
    try:
        import ipywidgets

        return True
    except ImportError:
        return False


def display_evaluation_comparison(before, after):
    print("current performance:")
    print(before)
    print()
    print("after dropping:")
    print(after)
    print()
    print("error reduction (higher is better):")
    print(before - after)
    print()


def xy_df(df, target, train_index=None):
    X = df.copy().drop(columns=target)
    y = df[target]

    if y.isna().sum():
        raise Exception(
            f'unable to use "{target}" as target column since it contains null values'
        )

    return X, y


## why is this needed?
def xy_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_index=None,
    test_size: Union[float, int] = None,
    random_state=None,
    stratify: Union[str, pd.Series] = None,
):
    if train_index != None and test_size != None:
        raise ValueError("cannot specify both train_index and test_size")
    if train_index != None and stratify != None:
        raise ValueError("cannot specify both train_index and stratify")

    test_size = default_value(test_size, 0.3)
    random_state = default_value(random_state, 314159)
    if stratify and isinstance(stratify, str):
        stratify = y if stratify == y.name else X[stratify]

    # print('test size: ', test_size)
    if train_index is None:
        # print('stratify', stratify)
        X_train, X_test, y_train, y_test = split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratify,
        )
    else:
        X_train, X_test, y_train, y_test = (
            X.loc[train_index],
            X.drop(index=train_index),
            y[train_index],
            y.drop(train_index),
        )
    return X_train, X_test, y_train, y_test


def fill_nulls_naively(df):
    # return df
    print(f"the following cols contain null values and will be filled naively:")
    nulls = df[df.columns[df.isna().sum().astype(bool)]]

    num_nulls = nulls.select_dtypes(include="number")
    print("numerical columns:")
    print(list(num_nulls.columns))
    num_nulls = num_nulls.fillna(num_nulls.mean())
    df[num_nulls.columns] = num_nulls

    cat_nulls = nulls.select_dtypes(exclude="number")
    print("categorical columns:")
    print(list(cat_nulls.columns))
    cat_nulls = cat_nulls.fillna(cat_nulls.mode())
    df[cat_nulls.columns] = cat_nulls

    return df.join(nulls.isna().add_suffix("_is_missing"))


def cleanup_df(df, target, train_index=None):
    X, y = xy_df(df=df, train_index=train_index, target=target)

    X = fill_nulls_naively(X)

    cat_cols = X.select_dtypes(exclude=["number"]).columns

    for col in cat_cols:
        print(f"casting {col} onto numerical values")
        X[col] = X[col].astype("category").cat.codes

    return X, y