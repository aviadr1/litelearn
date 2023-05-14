import catboost
import numpy as np

from litelearn.train_frame import TrainFrame
from litelearn.model_frame import ModelFrame
from litelearn.exceptions import TrainingException


def regress_df(
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
    use_categories=None,
    use_nulls=None,
):
    train_frame = TrainFrame.from_df(
        df=df,
        target=target,
        train_index=train_index,
        test_size=test_size,
        stratify=stratify,
        use_categories=use_categories,
        use_nulls=use_nulls,
    )
    if drop_columns:
        train_frame = train_frame.drop_columns(drop_columns)

    y = df[target]

    if not np.issubdtype(y.dtype, np.number):
        raise TrainingException(
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


def classify_df(
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
    use_categories=None,
    use_nulls=None,
):
    train_frame = TrainFrame.from_df(
        df=df,
        target=target,
        train_index=train_index,
        test_size=test_size,
        stratify=stratify,
        use_categories=use_categories,
        use_nulls=use_nulls,
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
    train_frame.cat_features = train_frame.X_train.columns.get_indexer(cats)

    train_frame.text_features = train_frame.X_train.select_dtypes(
        "object"
    ).columns.to_list()
    print("using text fields:", train_frame.text_features)

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
        cat_features=train_frame.cat_features,
        text_features=train_frame.text_features,
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
        cat_features=train_frame.cat_features,
        text_features=train_frame.text_features,
    )

    result = ModelFrame(
        model=model,
        train_frame=train_frame,
    )
    # result.eval_pool = eval_pool  # HACK!

    # result.display_evaluation()
    result.display_feature_importance()

    return result, eval_pool
