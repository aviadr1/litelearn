# %pip install --quiet --upgrade catboost googledrivedownloader shap altair

import logging
from copy import copy as copy_
from dataclasses import dataclass
from typing import *

# For transformations and predictions
import catboost
import numpy as np
import pandas as pd
import seaborn as sns
import shap

# For the tree visualization
from IPython.display import display
from pandas.api.types import is_numeric_dtype
from sklearn.inspection import permutation_importance

# For scoring
from sklearn.metrics import mean_squared_error as mse

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


@dataclass
class TrainFrame:
    """
    holds all the information needed to train a model
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    cat_features: List[int] = None
    text_features: List[str] = None
    dropped_features: List[str] = None
    df: pd.DataFrame = None
    segments: pd.DataFrame = None
    current_segments: dict = None

    def copy(self, deep=True):
        # TODO: is this taking a lot / too much memory?
        return TrainFrame(
            X_train=self.X_train.copy(deep=deep),
            y_train=self.y_train.copy(deep=deep),
            X_test=self.X_test.copy(deep=deep),
            y_test=self.y_test.copy(deep=deep),
            cat_features=copy_(self.cat_features),
            text_features=copy_(self.text_features),
            dropped_features=copy_(self.dropped_features),
            df=self.df.copy(deep=deep),
            segments=self.segments.copy(deep=deep)
            if self.segments is not None
            else None,
        )

    @staticmethod
    def from_df(
        df,
        target,
        train_index=None,
        test_size=None,
        stratify=None,
    ):
        X, y = cleanup_df(df=df, train_index=train_index, target=target)

        X_train, X_test, y_train, y_test = xy_split(
            X,
            y,
            train_index=train_index,
            test_size=test_size,
            stratify=stratify,
        )

        # TODO: what about cat_features and text_features, do we need this???

        train_frame = TrainFrame(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            df=df,
        )
        return train_frame

    def get_reduced_df(self, copy=True):
        if self.dropped_features:
            result = self.df.drop(columns=self.dropped_features, errors="ignore")
        else:
            result = self.df
        if copy:
            result = result.copy()

        return result

    def fit(
        self,
        iterations=None,
        logging_level=None,
        random_seed=None,
        learning_rate=None,
        early_stopping_rounds=None,
        plot=None,
        sample_weights=None,
        fit_kwargs={},
    ):
        # Text features not supported for regression
        print(self.text_features)
        if self.text_features is None:
            self.text_features = self.X_train.select_dtypes("object").columns.to_list()

        if len(self.text_features) > 0:
            print("dropping text fields:", self.text_features)
            return self.drop_columns(self.text_features).fit(
                iterations=iterations,
                logging_level=logging_level,
                random_seed=random_seed,
                learning_rate=learning_rate,
                early_stopping_rounds=early_stopping_rounds,
                plot=plot,
            )

        iterations = default_value(iterations, 1000)
        logging_level = default_value(logging_level, "Verbose")
        random_seed = default_value(random_seed, 42)
        # learning_rate  # default for learning_rate IS None
        early_stopping_rounds = default_value(early_stopping_rounds, 20)
        plot = default_value(plot, True if _is_interactive() else False)

        if self.cat_features is None:
            cats = self.X_train.select_dtypes("category").columns.to_list()
            print("using categories:", cats)
            self.cat_features = self.X_train.columns.get_indexer(cats)

        # TODO: support other metrics
        model = catboost.CatBoostRegressor(
            # eval_metric='MAE',
            # loss_function='MAE',
            eval_metric="RMSE",
            loss_function="RMSE",
            random_seed=random_seed,
            logging_level=logging_level,
            iterations=iterations,
            learning_rate=learning_rate,  # None by default, found 0.08 to be a good value
            early_stopping_rounds=early_stopping_rounds,
            cat_features=self.cat_features,
        )

        if sample_weights is not None:
            sample_weights = sample_weights.loc[self.X_train.index]

        visualizer = model.fit(
            self.X_train,
            self.y_train,
            eval_set=(self.X_test, self.y_test),
            #         logging_level='Verbose',  # you can uncomment this for text output
            #         logging_level='Debug',  # you can uncomment this for text output
            sample_weight=sample_weights,
            plot=plot,
            **fit_kwargs,
        )

        # eval_pool = catboost.Pool(X_test, y_test)

        result = ModelFrame(
            model=model,
            train_frame=self,
            # eval_pool=eval_pool,
        )
        return result

    def drop_columns(self, columns: List[str], whitelist: bool = False):
        """
        returns a new TrainFrame that excludes the given columns.
        it does so by removing the columns from X_train/X_test of the new frame.
        if @whitelist is True, then only the specified columns are retained

        Example:
            train_frame = TrainFrame(...)
            train_frame_smaller = train_frame.drop_columns(columns=['bad_feature', 'useless_id'])
            train_frame_smaller.get_df().to_csv('smaller.csv')
        """
        assert type(whitelist) is bool

        columns_set = set(columns)
        if whitelist:
            columns_set = set(self.X_train.columns) - columns_set
            columns = list(columns_set)

        cat_features, text_features = None, None
        if self.cat_features is not None:
            # cat features is a list of indices
            # easier to handle by just recalculating it
            cat_features = None
            for feature_i in self.cat_features:
                if self.X_train.iloc[:, feature_i].dtype.name != "category":
                    # TODO: handle case where cat_features contains
                    #       features with non category dtype
                    raise NotImplementedError()

        if self.text_features is not None:
            text_features = list(set(self.text_features) - columns_set)

        result = self.copy(deep=False)
        result.X_train = self.X_train.drop(columns=columns)
        result.X_test = self.X_test.drop(columns=columns)
        dropped_features = default_value(self.dropped_features, []) + columns
        return result

    def get_df(self):
        """
        returns a DataFrame that contains the train/test and y data
        """
        # TODO: test this implementation
        return pd.concat(
            [self.X_train.join(self.y_train), self.X_test.join(self.y_test)]
        )

    def get_stage_data(self, stage):
        if stage == "train":
            return self.X_train, self.y_train
        elif stage == "test":
            return self.X_test, self.y_test
        else:
            raise Exception("unknown stage", stage)


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


@dataclass
class ModelFrame:
    """
    Holds complete information about a trained model,
    and the TrainFrame that was used to train it
    """

    # dataclass attributes
    model: catboost.CatBoost
    train_frame: TrainFrame

    def __getattr__(self, item):
        """
        composite design pattern:
        ModelFrame acts as-if it is also a TrainFrame, and will return any attribute belonging
        to the train frame as if it also supports it.
        example:
            model_frame = ModelFrame(...)
            model_frame.X_train # actually returns model_frame.train_frame.X_train
            model.get_stage_data('train') # # actually returns model_frame.train_frame..get_stage_data('train')
        """

        # Gets called when the item is not found via __getattribute__
        return getattr(self.train_frame, item)  # search in the train_frame

    def copy(self, deep=True):
        return ModelFrame(
            model=self.model.copy() if deep else self.model,
            train_frame=self.train_frame.copy(deep=deep),
        )

    # not needed because of __getattr__
    # def get_stage_data(self, stage):
    #    return self.train_frame.get_stage_data(stage)

    def display_feature_importance(self, type=None):
        type = default_value(type, catboost.EFstrType.FeatureImportance)

        result = self.model.get_feature_importance(
            prettified=True,
            type=type,
            data=catboost.Pool(
                self.X_test,
                label=self.y_test,
                cat_features=self.cat_features,
                # text_features=self.text_features,
            ),
        )
        if self.current_segments is not None:
            s = ", ".join([f"{k}={v}" for k, v in self.current_segments.items()])
            result.columns = pd.MultiIndex.from_product([[s], result.columns])

        display(result.head(60))
        return result

    def set_segment(
        self,
        segment: Union[pd.Series, Callable, str],
        segment_name: Optional[str] = None,
    ):
        if callable(segment):
            if segment_name is None:
                raise ValueError(
                    "when creating a segment with a function, "
                    'you must set the "segment_name" parameter'
                )
            # apply function over all rows
            segment = self.df.apply(segment, axis=1).rename(segment_name)
        elif isinstance(segment, str):
            segment = self.df[segment]
        elif isinstance(segment, pd.Series):
            pass
        else:
            raise ValueError(
                "segment must be callable, str or series. not supported:", type(segment)
            )

        if segment.dtype.name != "category":
            segment = segment.astype("category")

        if not self.df.index.equals(segment.index):
            raise ValueError(
                f"the indexes don't match: {self.df.index} != {segment.index}"
            )
        if self.segments is None:
            self.segments = pd.DataFrame(index=self.df.index)

        self.segments[segment.name] = segment

    def get_segment_subset(
        self,
        segment_name: pd.Series,
        segment_value: str,
        X: Union[pd.DataFrame, pd.Series],
        y: Optional[pd.Series] = None,
    ):
        segment = self.segments[segment_name]
        # print(segment.head())
        # display(X)
        data_subset = segment.loc[X.index]
        subset = data_subset[data_subset == segment_value]
        if y is not None:
            return X.loc[subset.index], y.loc[subset.index]
        return X.loc[subset.index]

    def get_segmented_frame(
        self, segment_name: str, segment_value: str
    ) -> pd.DataFrame:
        result = self.copy(deep=False)  # shallow copy
        if result.current_segments is None:
            result.current_segments = {}
        result.current_segments[segment_name] = segment_value

        (
            result.train_frame.X_train,
            result.train_frame.y_train,
        ) = self.get_segment_subset(
            segment_name=segment_name,
            segment_value=segment_value,
            X=result.X_train,
            y=result.y_train,
        )
        result.train_frame.X_test, result.train_frame.y_test = self.get_segment_subset(
            segment_name=segment_name,
            segment_value=segment_value,
            X=result.X_test,
            y=result.y_test,
        )

        return result

    def display_evaluations(self, methods: Optional[Union[List[str], str]] = None):
        methods = default_value(methods, "all")
        methods = default_value(
            methods,
            [
                "metrics",
                "residuals",
                "permutation_importance",
                # 'shap',
            ],
            "all",
        )
        if isinstance(methods, str):
            methods = [methods]

        for method in methods:
            if method == "metrics":
                self.display_evaluation()
            elif method == "residuals":
                self.display_residuals()
            elif method == "permutation_importance":
                self.display_feature_importance()
            elif method == "shap":
                self.display_shap()
            else:
                raise ValueError(
                    f"{method} is not a valid value for an evaluation method"
                    "in {methods}"
                )

    def evaluate_segments(self, methods=None):
        methods = default_value(methods, ["metrics"])

        segment_names = []
        if self.segments is not None:
            segment_names = self.segments.columns.to_list()

        print("evaluating across segments:", segment_names)
        for segment_name in segment_names:
            for segment_value in self.segments[segment_name].cat.categories:
                frame = self.get_segmented_frame(segment_name, segment_value)
                frame.display_evaluations(methods=methods)

    def get_segmented_predictions(self):
        result = pd.DataFrame(index=self.df.index)
        if self.segments is not None:
            result = self.segments.copy()

        for stage in ["train", "test"]:
            X, y = self.get_stage_data(stage)
            if len(X) <= 0:
                continue

            y_pred = pd.DataFrame(self.model.predict(X)).set_index(y.index).squeeze()
            result.loc[y.index, "actual"] = y
            result.loc[y.index, "pred"] = y_pred
            result.loc[y.index, "stage"] = stage
            if is_numeric_dtype(y):
                result.loc[y.index, "error"] = y - y_pred
                result.loc[y.index, "abs_error"] = (y - y_pred).abs()
                result.loc[y.index, "sq_error"] = (y - y_pred) ** 2
            else:
                result.loc[y.index, "error"] = y != y_pred

        return result

    def get_evaluation(self):
        evaluation = pd.DataFrame()
        for stage in ["train", "test"]:
            X, y = self.get_stage_data(stage)
            if len(X) <= 0:
                continue

            y_pred = self.model.predict(X)
            # TODO: other metrics
            rmse = mse(y, y_pred, squared=False)
            evaluation.loc[stage, "rmse"] = rmse
            evaluation.loc[stage, "support"] = len(X)
        if self.current_segments:
            for name, value in self.current_segments.items():
                evaluation[name] = value
                evaluation = evaluation.set_index(name, append=True)
        evaluation["support"] = evaluation["support"].astype("int")
        return evaluation

    def display_shap(
        self,
        plot=None,
        max_display=None,
        exclude_features=None,
        stage=None,
    ):

        max_display = default_value(max_display, 20)
        stage = default_value(stage, "test")
        plot = default_value(plot, "summary")

        X, _ = self.get_stage_data(stage)

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        shap_values_df = pd.DataFrame(
            shap_values, columns=pd.Index(X.columns, name="features")
        )
        if exclude_features:
            shap_values_df = shap_values_df.drop(columns=exclude_features)
            X = X.drop(columns=exclude_features)

        if isinstance(max_display, float) and 0 < max_display <= 1:
            max_display = 1 + int(len(X.columns) * max_display)

        if plot == "summary":
            shap.summary_plot(shap_values_df.to_numpy(), X, max_display=max_display)
        elif plot == "cluster":
            strongest_features = (
                shap_values_df.abs().mean().sort_values(ascending=False)
            )
            strongest_features_index = strongest_features.head(max_display).index
            # strongest_features_index = shap_values_df.columns
            sns.clustermap(
                shap_values_df[strongest_features_index].corr().abs(),
                method="weighted",
                # cmap="coolwarm",
                figsize=(18, 18),
            )
        else:
            raise ValueError(f'{plot} is not a valid value for parameter "plot"')

    def display_evaluation(self):
        evaluation = self.get_evaluation()
        display(evaluation)

    def get_permutation_importance(self, n_repeats=None, random_state=None):
        n_repeats = default_value(n_repeats, 7)
        random_state = default_value(random_state, 1066)

        perm_importance = permutation_importance(
            self.model,
            self.X_test,
            self.y_test,  # self.y_test.astype('string'),
            n_repeats=n_repeats,
            random_state=random_state,
        )

        feature_importance = pd.DataFrame(
            {
                "mean": perm_importance.importances_mean,
                "std": perm_importance.importances_std,
            },
            index=self.X_test.columns,
        ).sort_values(by="mean", ascending=False)
        return feature_importance

    def display_permutation_importance(self, head=60, tail=20, n_repeats=None):
        feature_importance = self.get_permutation_importance(n_repeats=n_repeats)

        display(feature_importance.head(head))
        if len(feature_importance) > head:
            display(feature_importance.tail(min(len(feature_importance) - head, tail)))

    def display_residuals(self, stages=None):
        return residuals.display_residuals(self, stages=stages)

    def permutation_feature_selection(
        self,
        stds=1,
        threshold=0,
        k=5,
        n_repeats=None,  # number of repeats for importance calculation
        logging_level="Silent",
        plot=False,
        sample_weights=None,
    ):
        # see alternative implementation in
        # https://stackoverflow.com/questions/62537457/right-way-to-use-rfecv-and-permutation-importance-sklearn

        feature_importance = self.get_permutation_importance(n_repeats=n_repeats)
        feature_importance["upper_estimate"] = (
            feature_importance["mean"] + stds * feature_importance["std"]
        )
        worst = feature_importance[
            feature_importance.upper_estimate <= threshold
        ].sort_values(by="upper_estimate")
        if len(worst) == 0:
            raise StopIteration()

        # display(worst) # debug
        k_worst_feature_names = worst.head(k).index.to_list()
        print("dropping columns:")
        print(worst.head(k))
        print()
        new_model = self.train_frame.drop_columns(k_worst_feature_names).fit(
            logging_level=logging_level,
            plot=plot,
            sample_weights=sample_weights,
        )
        display_evaluation_comparison(self.get_evaluation(), new_model.get_evaluation())
        return new_model, k_worst_feature_names

    def progressive_permutation_feature_selection(
        self,
        stds=None,
        threshold=None,
        k=None,
        n_repeats=None,  # number of repeats for importance calculation
        logging_level="Silent",
        plot=False,
        sample_weights=None,
    ):
        stds = default_value(stds, [1, 0, -0.5])
        threshold = default_value(threshold, [0] * (len(stds) - 1) + [0])
        k = default_value(k, [5, 5, 3])

        result = []
        i = 0
        new_model = self
        result.append(new_model)

        for stds_, threshold_, k_ in zip(stds, threshold, k):
            print("std=", stds_)
            print("threshold=", threshold_)
            print("k=", k_)
            try:
                while True:
                    i += 1
                    (
                        new_model,
                        dropped_columns,
                    ) = new_model.permutation_feature_selection(
                        stds=stds_,
                        threshold=threshold_,
                        k=k_,
                        n_repeats=n_repeats,
                        logging_level=logging_level,
                        plot=plot,
                        sample_weights=sample_weights,
                    )

                    result.append(new_model)
            except StopIteration:
                pass
        display_evaluation_comparison(
            result[0].get_evaluation(), result[-1].get_evaluation()
        )

        return result

    def get_reduced_df(self, copy=True, orderby=None):
        result = self.train_frame.get_reduced_df(copy=copy)
        orderby = default_value(orderby, False)
        if orderby:  # reordering requested
            if orderby == "permutation_importance":
                # ordered from most to least important
                order = self.get_permutation_importance()
                # the target column will be first
                result = result.loc[:, [self.y_train.name] + order.index.to_list()]
            else:
                raise ValueError(
                    f'{orderby} is not a valid value for the parameter "orderby"'
                )

        return result


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


def core_regress_df(
    df,
    target,
    train_index=None,
    lr=None,
    iterations=None,
    test_size=None,
    sample_weights=None,
    fit_kwargs={},
    drop_columns=None,
    stratify=None,
):
    train_frame = TrainFrame.from_df(
        df=df,
        target=target,
        train_index=train_index,
        test_size=test_size,
        stratify=stratify,
    )
    if drop_columns:
        train_frame = train_frame.drop_columns(drop_columns)

    y = df[target]

    if not np.issubdtype(y.dtype, np.number):
        raise Exception(
            f'unable to use "{target}" as regression target since it is not numerical'
        )

    result = train_frame.fit(
        learning_rate=lr,
        iterations=iterations,
        fit_kwargs=fit_kwargs,
        sample_weights=sample_weights,
    )

    print(f"benchmark_stdev  {y.std():.4f}")
    result.display_evaluation()
    result.display_feature_importance()

    return result


def core_classify_df(
    df,
    target,
    train_index=None,
    lr=None,
    iterations=None,
    test_size=None,
    sample_weights=None,
    fit_kwargs={},
    drop_columns=None,
    stratify=None,
):
    train_frame = TrainFrame.from_df(
        df=df,
        target=target,
        train_index=train_index,
        test_size=test_size,
        stratify=stratify,
    )
    if drop_columns:
        train_frame = train_frame.drop_columns(drop_columns)
    y = df[target]

    print("Labels: {}".format(set(y)))
    # print('Zero count = {}, One count = {}'.format(len(y) - sum(y), sum(y)))
    print(y.value_counts())

    print("training")

    cats = train_frame.X_train.select_dtypes("category").columns.to_list()
    print("using categories:", cats)
    cats_index = train_frame.X_train.columns.get_indexer(cats)

    texts = train_frame.X_train.select_dtypes("object").columns.to_list()
    print("using text fields:", texts)

    model = catboost.CatBoostClassifier(
        custom_loss=["AUC", "Accuracy", "Precision", "Recall"],
        random_seed=42,
        # logging_level='Verbose',
        iterations=iterations,
        learning_rate=lr,
        early_stopping_rounds=20,
    )

    # TODO: move to TrainFrame
    visualizer = model.fit(
        train_frame.X_train,
        train_frame.y_train,
        cat_features=cats_index,
        text_features=texts,
        eval_set=(train_frame.X_test, train_frame.y_test),
        # logging_level='Verbose',  # you can uncomment this for text output
        # logging_level='Debug',  # you can uncomment this for text output
        plot=True,
        sample_weight=sample_weights.loc[train_frame.X_train.index]
        if sample_weights
        else None,
    )

    eval_pool = catboost.Pool(
        train_frame.X_test,
        train_frame.y_test,
        cat_features=cats_index,
        text_features=texts,
    )

    result = ModelFrame(
        model=model,
        train_frame=train_frame,
    )
    result.eval_pool = eval_pool  # HACK!

    # result.display_evaluation()
    result.display_feature_importance()

    return result, eval_pool
