import pytest
import seaborn as sns
import litelearn as ll


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
