from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Callable, Optional, List

import catboost
import litelearn
import pandas as pd
import seaborn as sns
import shap

# compatibility between IPython 8.7 and 7.9
try:
    from IPython.core.display_functions import display
except ImportError:
    from IPython.display import display

from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error as mse

# from litelearn import TrainFrame
from litelearn.expo_ds import (
    default_value,
    display_evaluation_comparison,
    UNKNOWN_CATEGORY_VALUE,
    NA_CATEGORY_VALUE
)
from . import residuals


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
        if item[:2] == "__" and item[-2:] == "__":
            raise AttributeError(item)  # we dont mess with dunders
        else:
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
                self.train_frame.X_test,
                label=self.train_frame.y_test,
                cat_features=self.train_frame.cat_features,
                # text_features=self.train_frame.text_features,
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
            X=result.train_frame.X_train,
            y=result.train_frame.y_train,
        )
        result.train_frame.X_test, result.train_frame.y_test = self.get_segment_subset(
            segment_name=segment_name,
            segment_value=segment_value,
            X=result.train_frame.X_test,
            y=result.train_frame.y_test,
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
            X, y = self.train_frame.get_stage_data(stage)
            if len(X) <= 0:
                continue

            y_pred = self.model.predict(X)
            # TODO: other metrics
            rmse = mse(y, y_pred, squared=False)
            evaluation.loc[stage, "rmse"] = rmse
            evaluation.loc[stage, "support"] = len(X)
        if self.train_frame.current_segments:
            for name, value in self.train_frame.current_segments.items():
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

        X, _ = self.train_frame.get_stage_data(stage)

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
            self.train_frame.X_test,
            self.train_frame.y_test,  # self.y_test.astype('string'),
            n_repeats=n_repeats,
            random_state=random_state,
        )

        feature_importance = pd.DataFrame(
            {
                "mean": perm_importance.importances_mean,
                "std": perm_importance.importances_std,
            },
            index=self.train_frame.X_test.columns,
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
                result = result.loc[
                    :, [self.train_frame.y_train.name] + order.index.to_list()
                ]
            else:
                raise ValueError(
                    f'{orderby} is not a valid value for the parameter "orderby"'
                )

        return result

    def predict_prepare(self, df, drop_unused=True):
        X_train = self.train_frame.X_train
        model_columns = X_train.columns.to_list()

        # print(df.columns.to_list())
        if drop_unused:
            drop_columns = set(df.columns) - set(model_columns)
            print("dropping columns:", drop_columns)
            df2 = df.drop(columns=drop_columns)

        # reorder columns
        df2 = df2[model_columns]

        # recast columns
        for col in model_columns:
            src_dtype = df2[col].dtype
            dst_dtype = X_train[col].dtype
            if src_dtype != dst_dtype:
                src_col = df2[col]
                if dst_dtype.name == "category":
                    unknown_values = set(df[col]) - set(X_train[col].cat.categories)
                    src_col = src_col.astype('string').replace(
                        to_replace=unknown_values, value=UNKNOWN_CATEGORY_VALUE
                    )

                    # fillna for categories - catboost can't handle NaNs for categorical cols
                    if src_col.isna().any():
                        print('fillna for', col)
                        src_col = src_col.astype('string').fillna(NA_CATEGORY_VALUE)

                print("recasting", col, "from", src_dtype, "to", dst_dtype)
                df2[col] = src_col.astype(dst_dtype)

        # print(df2.columns.to_list())
        return df2

    def predict(self, df, drop_unused=True):
        result = self.model.predict(self.predict_prepare(df, drop_unused=drop_unused))
        if result.ndim == 1:
            return pd.Series(result, index=df.index)
        elif result.ndim == 2 and result.shape[1] == 1:
            return pd.Series(result[:, 0], index=df.index)
        else:
            return pd.DataFrame(result, index=df.index)

    def predict_proba(self, df, drop_unused=True):
        return pd.DataFrame(
            self.model.predict_proba(self.predict_prepare(df, drop_unused=drop_unused)),
            columns=list(self.model.classes_),
            index=df.index,
        )
