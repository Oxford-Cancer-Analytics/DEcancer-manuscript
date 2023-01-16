import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from pandas import Index
from pandas import Series
from sklearn.feature_selection import f_classif
from skrebate import MultiSURF
from statsmodels.stats.multitest import fdrcorrection as fdr

from ._types import Frames


class Univariate:
    """A class for performing Univariate feature selection."""

    def train_validation_univariate(
        self,
        train_df: DataFrame,
        train_idx: ndarray,
        validation_idx: ndarray,
        data: DataFrame,
        targets_train: Index,
        targets_valid: Index,
    ) -> tuple[DataFrame, DataFrame]:
        """A preprocessing method used on training data.

        Parameters
        ----------
        train_df
            The training dataframe.
        train_idx
            The indicies for the training data.
        validation_idx
            The indicies for the validation data.
        data
            The full, original, dataframe.
        targets_train
            The labels for training data.
        targets_valid
            The labels for validation data.

        Returns
        -------
            Training and validation data, post-feature selection.
        """
        _, p_values = self._run_f_classif(train_df.values, targets_train)
        rej, ind = fdr(p_values)

        del _, p_values
        indicies = [index for index, (r, i) in enumerate(zip(rej, ind)) if r]
        if not indicies:
            indicies = [index for index, (r, i) in enumerate(zip(rej, ind)) if not r]

        train = pd.concat(
            [
                pd.Series(targets_train),
                pd.DataFrame(
                    data=data.iloc[train_idx, indicies].values,
                    columns=data.iloc[train_idx, indicies].columns,
                ),
            ],
            axis=1,
        )

        validation = pd.concat(
            [
                pd.Series(targets_valid),
                pd.DataFrame(
                    data=data.iloc[validation_idx, indicies].values,
                    columns=data.iloc[validation_idx, indicies].columns,
                ),
            ],
            axis=1,
        )

        return train, validation

    def _run_f_classif(self, x_data: ndarray, y_data: Series) -> tuple[ndarray, ndarray]:
        return f_classif(x_data, y_data)


class Multisurf:
    """A class for performing MultiSURF feature selection."""

    def train_validation_multisurf(
        self,
        train_df: DataFrame,
        train_idx: ndarray,
        validation_idx: ndarray,
        data: DataFrame,
        targets_train: Index,
        targets_valid: Index,
    ) -> tuple[DataFrame, DataFrame]:
        """A preprocessing method used on training data.

        Parameters
        ----------
        train_df
            The training dataframe.
        train_idx
            The indicies for the training data.
        validation_idx
            The indicies for the validation data.
        data
            The full, original, dataframe.
        targets_train
            The labels for training data.
        targets_valid
            The labels for validation data.

        Returns
        -------
            Training and validation data, post-feature selection.
        """
        print(f"Number of features before MultiSURF: {train_df.shape[1]}")
        features = self._run_multisurf(train_df, targets_train)

        indicies = [tf for tf in features.top_features_ if features.feature_importances_[tf] > 0]
        if not indicies:
            indicies = features.top_features_
        del features

        print(f"Number of features after MultiSURF: {len(indicies)}")

        # Get the top 80 features if there are more than 80
        if len(indicies) > train_df.shape[0]:
            indicies = indicies[:80]

        train = pd.concat(
            [
                pd.Series(targets_train),
                pd.DataFrame(
                    data=data.iloc[train_idx, indicies].values,
                    columns=data.iloc[train_idx, indicies].columns,
                ),
            ],
            axis=1,
        )

        validation = pd.concat(
            [
                pd.Series(targets_valid),
                pd.DataFrame(
                    data=data.iloc[validation_idx, indicies].values,
                    columns=data.iloc[validation_idx, indicies].columns,
                ),
            ],
            axis=1,
        )
        return train, validation

    def _run_multisurf(self, x_data: DataFrame, y_data: Index) -> MultiSURF:
        fs = MultiSURF(n_features_to_select=x_data.shape[1], n_jobs=-1)

        return fs.fit(x_data.values, y_data.values)


class KDEMultiSurf(Multisurf):
    """A class used for post-KDE Multisurf feature selection."""

    def post_kde(self, frames: Frames, x_valid: DataFrame, y_valid: Series) -> tuple[Frames, DataFrame, Series]:
        """Generates a new feature set from the augmented data samples.

        Parameters
        ----------
        frames
            The set of preprocessed data.
        x_valid
            The validation data.
        y_valid
            The labels for validation data.

        Returns
        -------
            The processed Frame data and validation values.
        """
        targets = frames.data["classification"].astype("category")
        kde_frame = frames.data.drop("classification", axis=1)

        # Have to run through this for each key because the values to fit
        # Multisurf are different
        features = self._run_multisurf(kde_frame, targets)
        indicies = [tf for tf in features.top_features_ if features.feature_importances_[tf] > 0]
        del features

        # Get the top 80 features if there are more than 80
        if len(indicies) > 80:
            indicies = indicies[:80]

        kde_frame = pd.concat(
            [
                pd.Series(targets),
                pd.DataFrame(
                    data=kde_frame.iloc[:, indicies].values,
                    columns=kde_frame.iloc[:, indicies].columns,
                ),
            ],
            axis=1,
        )

        validation = pd.DataFrame(
            data=x_valid.iloc[:, indicies].values,
            columns=x_valid.iloc[:, indicies].columns,
        )

        return Frames(frames.name, kde_frame), validation, y_valid


class IntersectedFeatureSelector(Multisurf, Univariate):
    """A class for performing MultiSURF and Univariate feature selection.

    The index values which are common in both MultiSURF and Univariate
    feature selection.
    """

    def __init__(self) -> None:
        self.original_multisurf: MultiSURF | None = None
        self.ranked: list[int] = []
        self.ranked_final: list[int] = []
        self.unordered_final: list[int] = []

    def train_validation_both_fs(
        self,
        train_df: DataFrame,
        train_idx: ndarray,
        validation_idx: ndarray,
        data: DataFrame,
        targets_train: Index,
        targets_valid: Index,
    ) -> tuple[DataFrame, DataFrame]:
        """A preprocessing method used on training data.

        Parameters
        ----------
        train_df
            The training dataframe.
        train_idx
            The indicies for the training data.
        validation_idx
            The indicies for the validation data.
        data
            The full, original, dataframe.
        targets_train
            The labels for training data.
        targets_valid
            The labels for validation data.

        Returns
        -------
            Training and validation data, post-feature selection.
        """
        _, p_values = self._run_f_classif(train_df.values, targets_train)
        rej, ind = fdr(p_values)
        # For all the True values after p_value adjustment, return the
        # index of the variable
        uni_indicies = {index for index, (r, i) in enumerate(zip(rej, ind)) if r}

        # If there are no true values, then take them all
        if not uni_indicies:
            uni_indicies = {index for index, (r, i) in enumerate(zip(rej, ind)) if not r}

        # Returns the feature indicies of uni_indicies after applying
        # MultiSURF and selecting positive values
        multi = self._run_multisurf(train_df.iloc[:, list(uni_indicies)], targets_train)
        multi_adjusted_indicies = [tf for tf in multi.top_features_ if multi.feature_importances_[tf] > 0]
        multi_indicies = [list(uni_indicies)[i] for i in multi_adjusted_indicies]
        del multi, _, p_values, rej, ind

        # Common feature index values
        final_indicies = list(uni_indicies.intersection(multi_indicies))

        # Guard to check for each fold, if the features change
        if not self.original_multisurf or (set(final_indicies) != set(self.unordered_final)):
            self.unordered_final = final_indicies
            self.original_multisurf = self._run_multisurf(train_df, targets_train)
            self.ranked = sorted(
                i for ui in uni_indicies for i, tf in enumerate(self.original_multisurf.top_features_) if tf == ui
            )
            self.ranked_final = sorted(
                i for ui in final_indicies for i, tf in enumerate(self.original_multisurf.top_features_) if tf == ui
            )

        # Preserve the order of the final_indicies based on the order of
        # the original_multisurf.top_features_
        final_indicies = [
            fi
            for o in self.ranked_final
            for fi in self.unordered_final
            if self.original_multisurf.top_features_[o] == fi
        ]
        del uni_indicies, multi_indicies

        # Use the top 80 if there are more than 80 features
        if len(final_indicies) > train_df.shape[0]:
            final_indicies = final_indicies[:80]

        # Classification first so that we can easily remove the last
        # variablewhich is also the least important
        train = pd.concat(
            [
                pd.Series(targets_train),
                pd.DataFrame(
                    data=data.iloc[train_idx, final_indicies].values,
                    columns=data.iloc[train_idx, final_indicies].columns,
                ),
            ],
            axis=1,
        )

        validation = pd.concat(
            [
                pd.Series(targets_valid),
                pd.DataFrame(
                    data=data.iloc[validation_idx, final_indicies].values,
                    columns=data.iloc[validation_idx, final_indicies].columns,
                ),
            ],
            axis=1,
        )

        return train, validation
