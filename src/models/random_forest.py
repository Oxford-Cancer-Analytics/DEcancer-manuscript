from typing import Any
from typing import cast
from typing import Mapping
from typing import Sequence

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from src.cancerseek.constants import ComboParams
from src.data_models import ComboOptions
from src.data_models import ConstantOptions

from .._types import Frames
from ..parameters import Parameter
from .base import Model


class RandomForest(Model):
    """A class for the Random Forest which extends the Model class."""

    __slots__ = (
        "data",
        "collection",
        "preprocessor",
        "train_info",
        "test_info",
        "clf",
        "proteins_loaded",
        "combs",
        "hyperparameters",
    )

    def __init__(
        self,
        data: DataFrame,
        /,
        combo_options: ComboOptions | ComboParams,
        constant_options: ConstantOptions | None,
        params: Sequence[Parameter] = [Parameter(RandomForestClassifier)],
    ):
        super().__init__(data)

        self.data = data
        self.clf = RandomForestClassifier()
        self.proteins_loaded = False
        self.combs: dict[str, dict[int, list[str]]] = {
            "imbalanced_early": {},
            "balanced": {},
            "imbalanced_healthy": {},
        }
        self.combo_options = combo_options
        self.constant_options = constant_options
        self.collection = combo_options.collection
        self.preprocessor = combo_options.preprocessor
        self.hyperparameters = self._generate_hyperparameters(params)

    def preprocess(
        self, train_idx: np.ndarray, validation_idx: np.ndarray, **kwargs: object
    ) -> tuple[Frames, DataFrame, Series]:
        """Data preprocessing of both training and validation data.

        Parameters
        ----------
        train_idx
            The indices from a split generator.

        validation_idx
            The indices from a split generator.

        **kwargs
            Extra arguments passed as flags. Includes `multisurf`,
            `uni_multi`, `no_preprocessing` or `both_fs` as
            preprocessing and `kde` as KDE application.

            Default of `both_fs`: True, `kde`: True.

        Returns
        -------
        Frames
            A NamedTuple of KernelDensity values or a dataframe of
            training data.

        DataFrame
            The X_validation values or None.

        Series
            The Y_validation labels.
        """
        options = self._validate_kwargs(**kwargs)

        train_df = self.data.iloc[train_idx, :]
        validation_df = self.data.iloc[validation_idx, :]
        targets_train = Series(train_df.index.get_level_values(level="classification").astype("category"))
        targets_valid = Series(validation_df.index.get_level_values(level="classification").astype("category"))

        if options["both_fs"]:
            train, validation = self.train_validation_both_fs(
                train_df,
                train_idx,
                validation_idx,
                self.data,
                targets_train,
                targets_valid,
            )
        elif options["multisurf"]:
            train, validation = self.train_validation_multisurf(
                train_df,
                train_idx,
                validation_idx,
                self.data,
                targets_train,
                targets_valid,
            )
        elif options["uni_multi"]:
            train, validation = self.train_validation_univariate(
                train_df,
                train_idx,
                validation_idx,
                self.data,
                targets_train,
                targets_valid,
            )
        else:
            train = pd.concat(
                [
                    pd.Series(targets_train),
                    pd.DataFrame(train_df.values, columns=train_df.columns),
                ],
                axis=1,
            )
            validation = pd.concat(
                [
                    pd.Series(targets_valid),
                    pd.DataFrame(validation_df.values, columns=validation_df.columns),
                ],
                axis=1,
            )

        del train_df, validation_df, targets_train, targets_valid

        x_valid = validation.drop("classification", axis=1)
        y_valid = validation["classification"].astype("category")

        frames = self.augment(train, options)

        # No longer needed at this point so can remove to free up memory.
        # Will be renewed on next MCCV iteration
        del validation, train

        return frames, x_valid, y_valid

    def train(
        self,
        frames: Frames,
        x_valid: DataFrame,
        y_valid: Series,
        index: int,
        **hyperparameters: Mapping[str, Any],
    ) -> None:
        """Training the Random Forest model.

        Parameters
        ----------
        frames
            A tuple of the Kernel Density name and the data.
        x_valid
            The X_validation values.
        y_valid
            The Y_validation labels.
        index
            The iteration of the MCCV fold.
        **hyperparameters
            Pairs of hyperparameter names and values.
        """
        if isinstance(hyperparameters["max_depth"], float):
            hyperparameters["max_depth"] = int(hyperparameters["max_depth"])  # type: ignore

        self.clf.set_params(**hyperparameters)
        if self.preprocessor == "uni_multi":
            self._train(frames.name, frames.data, index, x_valid, y_valid)
            return

        print(
            f"{self.collection} | {index}: Final feature set size {frames.data.shape} used for training {frames.name}."
        )

        x_data = frames.data.drop("classification", axis=1).astype("float16")
        y_data = frames.data["classification"].astype("category").values

        # Training
        estimator = self.clf.fit(x_data, y_data)
        y_score = estimator.predict_proba(x_valid)

        if isinstance(self.combo_options, ComboOptions) and isinstance(self.constant_options, ConstantOptions):
            auc = metrics.roc_auc_score(y_valid, y_score[:, 1])
            inner_dict: dict[str, list[float | str]] = (
                {"auc": []} if not self.constant_options.feature_selection else {"auc": [], "proteins": []}
            )
            self.train_info[frames.name][index] = inner_dict
            self.train_info[frames.name][index]["auc"].append(auc)
            if self.constant_options.feature_selection:
                self.train_info[frames.name][index]["proteins"].extend(x_data.columns)
        else:
            y_truth_onehot = np.zeros([y_valid.shape[0], int(np.max(y_valid) - np.min(y_valid) + 1)])
            y_truth_onehot[np.arange(y_valid.shape[0]), y_valid.astype(int)] = 1

            self.train_info["feature_importances"] = self.clf.feature_importances_
            self.train_info["auc"] = metrics.roc_auc_score(y_truth_onehot, y_score)
            self.train_info["y_truth"] = y_truth_onehot
            self.train_info["y_pred"] = y_score

    def train_rfe(
        self,
        frames: Frames,
        x_valid: DataFrame,
        y_valid: Series,
        index: int,
        **hyperparameters: Mapping[str, Any],
    ) -> None:
        """Trains a recursive Random Forest model.

        Parameters
         ----------
        frames
            A tuple of the Kernel Density name and the data.
        x_valid
            The X_validation values.
        y_valid
            The Y_validation labels.
        index
            The iteration of the MCCV fold.
        **hyperparameters
            Pairs of hyperparameter names and values.
        """
        if isinstance(hyperparameters["max_depth"], float):
            hyperparameters["max_depth"] = int(hyperparameters["max_depth"])  # type: ignore

        self.clf.set_params(**hyperparameters)
        print(f"MCCV {index}")

        x_data = frames.data.drop("classification", axis=1).astype("float16")
        y_data = frames.data["classification"].astype("category").values

        # Training
        estimator = self.clf.fit(x_data, y_data)
        y_score = estimator.predict_proba(x_valid)[:, 1]

        # Scoring
        auc = metrics.roc_auc_score(y_valid, y_score)
        if not self.proteins_loaded:
            self.train_info[frames.name] = {"proteins": []}
            self.train_info[frames.name]["proteins"].extend(
                list(c for c in frames.data.columns if c != "classification")
            )
            self.proteins_loaded = True

        self.train_info[frames.name][index] = {"auc": [], "feature_importances": []}
        self.train_info[frames.name][index]["auc"].append(auc)
        self.train_info[frames.name][index]["feature_importances"].extend(list(estimator.feature_importances_))

    def train_testing(self, frames: Frames, x_test: DataFrame, y_test: Series) -> float:
        """Trains a Random Forest model.

        Parameters
        ----------
        frames
            A tuple of the Kernel Density name and the data.
        x_test
            The x_test values.
        y_test
            The y_test values.

        Returns
        -------
            An AUC score.
        """
        x_data = frames.data.drop("classification", axis=1).astype("float16")
        y_data = frames.data["classification"].astype("category").values

        # Training
        estimator = self.clf.fit(x_data.values, y_data)
        y_score = estimator.predict_proba(x_test.values)[:, 1]

        # Scoring
        auc = cast(float, metrics.roc_auc_score(y_test, y_score))
        return auc

    def _train(
        self,
        aug_key: str,
        kde_frame: DataFrame,
        index: int,
        x_valid: DataFrame,
        y_valid: Series,
    ) -> None:
        self.combs[aug_key][index] = [c for c in list(kde_frame.columns) if c != "classification"]
        self.train_info[aug_key][index] = []

        x_data = kde_frame.drop("classification", axis=1).astype("float16").values
        y_data = kde_frame["classification"].astype("category").values

        estimator = self.clf.fit(x_data, y_data)
        y_score = estimator.predict_proba(x_valid.values)[:, 1]

        auc = metrics.roc_auc_score(y_valid, y_score)
        self.train_info[aug_key][index].append(auc)
