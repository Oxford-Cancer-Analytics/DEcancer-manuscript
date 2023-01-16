from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Final
from typing import Literal
from typing import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.models.random_forest import RandomForest
from src.parameters import Parameter

from .constants import ComboParams
from .constants import KDEMethod
from .constants import PipelineMode
from .utility import Augmentation

if TYPE_CHECKING:
    from .._types import ModelEstimator


@dataclass
class DataSplit:
    """Holds all of the splits of data."""

    ids_train: np.ndarray
    ids_test: np.ndarray
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    y_source_test: np.ndarray
    y_stage_test: np.ndarray


# Class implementing K-Fold and Monte Carlo cross validation
class ValidationMethod:
    """A class to determine the cross-validation method."""

    def __init__(self, seed: int, folds: int = 1) -> None:
        self.seed = seed
        self.folds = folds
        self.valid_size = 0.2

    def cross_validation_split(self, valid_method: Literal["K_FOLD", "MONTE_CARLO"]) -> ShuffleSplit | StratifiedKFold:
        """Splits the data according to the `valid_method`.

        Parameters
        ----------
        valid_method
            The type of cross-validation to apply.

        Returns
        -------
            An instance of a cross-validation split.
        """
        # Monte Carlo Cross Validation
        if valid_method == "MONTE_CARLO":
            return ShuffleSplit(
                n_splits=self.folds,
                test_size=self.valid_size,
                random_state=self.seed,
            )
        # Stratified K-Fold Cross Validation
        else:
            # random_state should be its default if shuffle=False
            return StratifiedKFold(
                n_splits=self.folds,
            )


class ParameterSearch:
    """Generation of hyperparameters for each model to train and test.

    Hyperparameters are input from a combination pipeline which are then
    transformed so they can be set for each model. The data is split,
    scaled and trained and tested for each pipeline combination.
    """

    def __init__(
        self,
        model: type[ModelEstimator],
        parameters: Sequence[Parameter],
        pipeline_type: PipelineMode,
        combo_options: ComboParams,
        preprocessing_params: dict[str, KDEMethod] = {},
        test_size: float = 0.2,
        valid_method: Literal["K_FOLD", "MONTE_CARLO"] = "MONTE_CARLO",
        seed: int = 42,
    ):
        self.model = model
        self.hyperparameters = parameters
        self.pipeline_type = pipeline_type
        self.combo_options = combo_options
        self.preprocessing_params = preprocessing_params
        self.test_size = test_size
        self.valid_method: Final = valid_method
        self.seed = seed

        # Choose which cross validation method to use: MonteCarlo vs KFold
        self.cv_split = ValidationMethod(self.seed).cross_validation_split(self.valid_method)

    def init_data(
        self,
        sample_ids: np.ndarray,
        x_data: np.ndarray,
        y_data: np.ndarray,
        y_source: np.ndarray,
        y_stage: np.ndarray,
    ) -> None:
        """Sets instance variables for the data.

        Parameters
        ----------
        sample_ids
            The sample IDs for each patient.
        x_data
            The original x data.
        y_data
            The y labels for the `x_data`.
        y_source
            The y source data.
        y_stage
            The y cancer stage data.
        """
        self.sample_ids = sample_ids
        self.x_data = x_data
        self.y_data = y_data
        self.y_source = y_source
        self.y_stage = y_stage

    def run_search(self) -> list[dict[str, Any]]:
        """Selection of hyperparameters and data splitting.

        This is used as input for running the training and/or testing
        pipelines.

        Returns
        -------
            A list of results including the model metrics, data and
            labels used.
        """
        np.random.seed(self.seed)

        # Stratify with respect to class and stage of cancer
        self.stratify_label = list(zip(map(str, self.y_source), map(str, self.y_stage)))

        (ids_train, ids_test, x_train, x_test, y_train, y_test, _, y_source_test, _, y_stage_test,) = train_test_split(
            self.sample_ids,
            self.x_data,
            self.y_data,
            self.y_source,
            self.y_stage,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=self.stratify_label,
        )

        # Scale the data to reasonable bounds
        x_scaler: StandardScaler = StandardScaler().fit(x_train)
        x_train = x_scaler.transform(x_train)
        x_test = x_scaler.transform(x_test)

        # Well now let's just do the search across the hyperparameter space
        data_bundle = DataSplit(
            ids_train,
            ids_test,
            x_train,
            x_test,
            y_train,
            y_test,
            y_source_test,
            y_stage_test,
        )
        results = self._parameter_search(data_bundle)

        return results

    def _parameter_search(self, data: DataSplit) -> list[dict[str, Any]]:
        results = []
        model = self.model(pd.DataFrame(self.x_data), self.combo_options, None, self.hyperparameters)
        for param_set in model.hyperparameters:
            kde_params = self.preprocessing_params["kde"]
            if self.pipeline_type == PipelineMode.TESTING:
                print("\nFinally... testing the model")
                assert isinstance(model, RandomForest)

                train_frame = Augmentation().augment_train_set(
                    data.x_train,
                    data.y_train,
                    kde_params,
                    self.seed,
                )

                model.clf.set_params(**param_set)
                model.train_testing(train_frame, pd.DataFrame(data.x_test), pd.Series(data.y_test))

                feature_importances = model.train_info.get("feature_importances")
                roc_auc = model.train_info.get("auc")

                info = {
                    "train_ids": data.ids_train,
                    "test_ids": data.ids_test,
                    "test_results": model.train_info,
                    "source_test_labels": data.y_source_test,
                    "stage_test_labels": data.y_stage_test,
                }

                results.append(
                    {
                        "results": info,
                        "feature_importances": feature_importances,
                        "auc": roc_auc,
                        "params": param_set,
                        "preprocess": self.preprocessing_params,
                    }
                )
            else:
                print(f"Current parameters: {param_set}")
                folds = []

                for index, (train_set, validation_set) in enumerate(self.cv_split.split(data.x_train, data.y_train)):
                    print(f"\nCurrent MCCV Fold: {index}")
                    # Train
                    train_data = data.x_train[train_set]
                    train_label = data.y_train[train_set]

                    # Validation
                    valid_data = data.x_train[validation_set]
                    valid_label = data.y_train[validation_set]

                    train_frame = Augmentation().augment_train_set(
                        train_data,
                        train_label,
                        kde_params,
                        self.seed,
                    )

                    # Once we have our model and data, train all the model
                    model.train(train_frame, pd.DataFrame(valid_data), pd.Series(valid_label), index, **param_set)

                    folds.append(
                        {
                            "train_results": model.train_info,
                            "source_test_labels": data.y_source_test,
                            "stage_test_labels": data.y_stage_test,
                        }
                    )

                if self.model == RandomForest:
                    feature_importances = np.mean(
                        [fold["train_results"]["feature_importances"] for fold in folds],
                        axis=0,
                    )
                else:
                    feature_importances = None
                auc_data = [fold["train_results"]["auc"] for fold in folds]
                roc_auc = np.mean(auc_data, axis=0)
                auc_sem = np.std(auc_data) / np.sqrt(np.size(auc_data))
                results.append(
                    {
                        "results": folds,
                        "feature_importances": feature_importances,
                        "auc": roc_auc,
                        "auc_sem": auc_sem,
                        "params": param_set,
                        "preprocess": self.preprocessing_params,
                    }
                )
        return results
