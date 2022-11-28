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
