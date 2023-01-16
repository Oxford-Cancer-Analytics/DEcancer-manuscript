from typing import cast
from typing import Mapping
from typing import TypeAlias

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import model_selection
from sklearn.neighbors import KernelDensity

from ._types import Frames
from ._types import KDEType
from .constants import AugmentationMethod
from .constants import Constants


Ratio: TypeAlias = tuple[int, int]


def ratio(input_size: int, distribution: tuple[int, int] = (1, 1), size: int = 5) -> Ratio:
    """Generates a ratio based on the given distribution.

    Parameters
    ----------
    input_size
        Original number of samples.
    distribution, optional
        Proportion of samples to draw, by default (1, 1)
    size, optional
        Augmentation factor, by default 5

    Returns
    -------
        The new size based on the distribution.
    """
    dist = input_size * size

    n, d = distribution

    try:
        ratio_n = int((dist / (n + d)) * n)
        ratio_d = int((dist / (n + d)) * d)
    except ZeroDivisionError:
        ratio_n, ratio_d = input_size, input_size

    return ratio_n, ratio_d


class KDE:
    """A class for performing Kernel Density Estimation."""

    @staticmethod
    def _grid_search(df: DataFrame) -> KernelDensity:
        """Bandwidth selection using a GridSearchCV.

         Parameters
         ----------
         df
            The dataframe used to fit the gridsearch.

        Returns
        -------
            The best estimator from the Kernel Density.
        """
        params = {"bandwidth": np.logspace(-1, 1, 100)}

        grid = model_selection.GridSearchCV(
            KernelDensity(algorithm="auto", kernel="gaussian", metric="euclidean"),
            params,
            n_jobs=1,
        )

        grid.fit(df.drop("classification", axis=1))

        return grid.best_estimator_  # type: ignore

    @staticmethod
    def _augmentation(
        df: DataFrame,
        kde_early: KernelDensity,
        kde_healthy: KernelDensity,
        ratio_samples: Ratio,
    ) -> DataFrame:
        """Augmentation of the Kernel Density Estimator.

        Parameters
        ----------
        df
            The original training dataframe.

        kde_early
            The best estimator from the fitted early cancer dataset.

        kde_healthy
            The best estimator from the fitted healthy dataset.

        ratio_samples
            A tuple of samples to draw.

        Returns
        -------
            The augmented dataframe of the original data plus the
            synthetic data.
        """
        early_samples, healthy_samples = ratio_samples
        df.columns = [
            "classification",
            *[c for c in df.columns if c != "classification"],
        ]

        early_sampling = pd.DataFrame(
            kde_early.sample(n_samples=early_samples, random_state=Constants.RANDOM_STATE.value)
        ).astype("float16")
        early = pd.Series(["Early"] * len(early_sampling))
        early_sampling.insert(0, "classification", early)
        early_sampling.columns = df.columns

        healthy_sampling = pd.DataFrame(
            kde_healthy.sample(n_samples=healthy_samples, random_state=Constants.RANDOM_STATE.value)
        ).astype("float16")
        healthy = pd.Series(["Healthy"] * len(healthy_sampling))
        healthy_sampling.insert(0, "classification", healthy)
        healthy_sampling.columns = df.columns

        return pd.concat([early_sampling, healthy_sampling], ignore_index=True)

    def augment(self, df: DataFrame, options: Mapping[str, bool | tuple[int, int]]) -> Frames:
        """Applies Kernel Density to draw synthetic samples from.

        Parameters
        ----------
        df
            The reduced DataFrame.

        options
            The options to identify the KDEType.

        Returns
        -------
            A set of preprocessed data by Kernel Density Augmentation.
        """
        (kde_name,) = cast(
            list[KDEType],
            list(filter(lambda x: not isinstance(options[x], bool), options)),
        )

        distribution = options[kde_name]
        assert type(distribution) is tuple

        if distribution == AugmentationMethod.NONE.value:
            kde_data = df
        else:
            df_early = df.loc[df["classification"] == "Early", :]
            df_healthy = df.loc[df["classification"] == "Healthy", :]

            kde_early = self._grid_search(df_early)
            kde_healthy = self._grid_search(df_healthy)
            kde_data = self._augmentation(df, kde_early, kde_healthy, ratio(df.shape[0], distribution))  # noqa E501

        return Frames(name=kde_name, data=kde_data)
