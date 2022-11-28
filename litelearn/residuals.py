from __future__ import annotations

from typing import *

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# import litelearn.model_frame
from . import expo_ds


def display_residual_stage(
    self: litelearn.model_frame.ModelFrame,
    ax,
    stage,
    y,
    y_pred,
    ideal_line,
    ylabel,
    title,
):
    styles = {"train": "o", "test": "x"}
    line_colors = {"train": "salmon", "test": "teal"}
    marker_colors = {"train": "maroon", "test": "navy"}
    alpha = {"train": 0.25, "test": 0.99}

    sns.scatterplot(
        x=y,
        y=y_pred,
        ax=ax,
        marker=styles[stage],
        label=stage,
        alpha=alpha[stage],
        color=marker_colors[stage],
    )
    sns.regplot(
        x=y,
        y=y_pred,
        ax=ax,
        scatter=False,
        label=stage,
        line_kws={"color": line_colors[stage], "lw": 3},
    )
    ax.plot(ideal_line[0], ideal_line[1], "r--")

    ax.set_xlabel("actual")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return


def display_residuals(
    self: litelearn.model_frame.ModelFrame,
    stages: Optional[Union[str, List[str]]] = None,
):
    stages = expo_ds.default_value(stages, ["train", "test"])
    is_combine = stages == "combine"

    height = 8
    if not is_combine:
        height *= 2
    plt.figure(figsize=(16, height))

    if is_combine:
        stages_list = ["train", "test"]
    else:
        if isinstance(stages, str):
            stages_list = [stages]
        elif isinstance(stages, list):
            stages_list = stages
        else:
            raise ValueError(f'{stages} is not valid value for the "stages" parameter')

    for i, stage in enumerate(stages_list):
        plot_i = np.array([1, 2, 1])
        if not is_combine:
            plot_i = np.array([2, 2, 1 + i * 2])

        X, y = self.get_stage_data(stage)
        y_pred = self.model.predict(X)

        band = 0.1 * (y.max() - y.min())
        min_, max_ = y.min() - band, y.max() + band

        ax = plt.subplot(*plot_i)
        display_residual_stage(
            self=self,
            ax=ax,
            stage=stage,
            y=y,
            y_pred=y_pred,
            ideal_line=([min_, max_], [min_, max_]),
            ylabel="predicted",
            title="predicted/actual",
        )

        plot_i += [0, 0, 1]  # next column
        ax = plt.subplot(*plot_i)
        display_residual_stage(
            self=self,
            ax=ax,
            stage=stage,
            y=y,
            y_pred=y_pred - y,
            ideal_line=([min_, max_], [0, 0]),
            ylabel="residual $pred - actual$",
            title="predicted/actual",
        )

    if self.current_segments:
        title = ", ".join([f"{k}={v}" for k, v in self.current_segments.items()])
        plt.suptitle(title)

    plt.legend()
    plt.tight_layout()
