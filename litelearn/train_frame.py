from copy import copy as copy_
from dataclasses import dataclass
from typing import List

import catboost
import pandas as pd

from litelearn import ModelFrame
from litelearn.expo_ds import cleanup_df, xy_split, default_value, _is_interactive


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

        try:
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
        except catboost.CatboostError as err:
            display(self.X_train.info())
            raise

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
