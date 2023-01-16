from __future__ import annotations

import pprint
from itertools import product
from typing import Sequence

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .._types import BaseEstimator
from .._types import DataTypes
from .._types import Models
from .._types import Pipelines
from .._types import Preprocessors
from ..constants import AugmentationMethod
from ..data import control_data
from ..data import get_data
from ..data_models import ComboOptions
from ..data_models import ConstantOptions
from ..data_models import Data
from ..parameters import Parameter
from .biological import Biological
from .non_biological import NonBiological

pp = pprint.PrettyPrinter(indent=2, width=200, compact=True, sort_dicts=False)


class Pipeline:
    """The main pipeline class.

    This is used for the main functionality with composition to the
    biological and non_biological classes.
    """

    def __init__(
        self,
        combo_options: ComboOptions,
        constant_options: ConstantOptions = ConstantOptions(),
        hyperparameters: Sequence[Parameter] = [Parameter(RandomForestClassifier)],
        data: list[dict] | None = None,
    ) -> None:
        self.hyperparameters = hyperparameters
        self.data = data
        self.__post_init()

        self.non_biological = NonBiological(
            self.model_name,  # type: ignore
            combo_options,
            constant_options,
            hyperparameters,
            data=data,
        )

        self.biological = Biological(
            self.model_name,  # type: ignore
            combo_options,
            constant_options,
            hyperparameters,
        )

    def __post_init(self) -> None:
        model_dict: dict[BaseEstimator, Models] = {
            RandomForestClassifier: "random_forest",
            LogisticRegression: "logistic_regression_l2",
            MLPClassifier: "multi_layer_perceptron",
            SVC: "support_vector_machine",
            DecisionTreeClassifier: "decision_tree",
            XGBClassifier: "xgboost",
        }  # type: ignore

        if self.hyperparameters:
            classifier = next(iter(self.hyperparameters)).model
            self.model_name = model_dict[classifier]  # type: ignore


def combo_pipelines(
    collections: Sequence[str],
    kde_methods: Sequence[AugmentationMethod] = [AugmentationMethod.NONE],
    preprocessors: Sequence[Preprocessors] = ["no_preprocessing"],
    pipeline_types: Sequence[Pipelines] = ["non_biological"],
    data_types: Sequence[DataTypes] = ["all"],
    *,
    constant_options: ConstantOptions = ConstantOptions(),
    hyperparameters: Sequence[Parameter] = [Parameter(RandomForestClassifier)],
    data: list[dict] | None = None,
) -> Sequence[Biological | NonBiological]:
    """Produces the pipeline combinations based on the inputs.

    Parameters
    ----------
    collections, optional
        A combination of nanoparticles, by default ["DP"]
    kde_methods, optional
        A combination of augmentation methods, by default
        [AugmentationMethod.NONE]
    preprocessors, optional
        A combination of prepocessing methods to apply, by default
        ["no_preprocessing"]
    pipeline_types, optional
        A combination of pipeline types, by default ["non_biological"]
    data_types, optional
        Whether to use all or control data, by default ["all"]
    constant_options, optional
        The set of options which are constant, by default ConstantOptions()
    hyperparameters, optional
        The hyperparameters to use for each combination, by default
        [Parameter(RandomForestClassifier)]
    data, optional
        A list of data to use for the non_biological pipeline, by default
        None

    Returns
    -------
        A sequence of pipeline combinations.

    Raises
    ------
    NotImplementedError
        If the `data_type` or `pipeline_type` is not valid.
    """
    pipelines_instances = []
    for combos in product(collections, kde_methods, preprocessors, pipeline_types, data_types):
        combo = ComboOptions(*combos)

        # Control data doesn't have any diluted plasma proteins
        if (combo.collection == "DP") and (combo.data_type == "control"):
            continue  # pragma: no cover

        # We currently only use the biological pipeline with no kde
        if (combo.pipeline_type == "biological") and (combo.kde_method.value != (0, 0)):
            continue  # pragma: no cover

        match combo.data_type:
            case "control":
                file_data = dict(control_data())[combo.collection]
                constant_options.control_data = True
            case "all":
                file_data = dict(get_data())[combo.collection]
                constant_options.control_data = False
            case _:
                raise NotImplementedError("Make sure your parameters are a sequence.")

        # Add the data to be used when looping
        partition_data = Data(
            file_data[0].astype("float16"), file_data[1].astype("float16"), pd.concat(file_data).astype("float16")
        )
        combo.data = partition_data

        pipeline = Pipeline(combo, constant_options, hyperparameters, data=data)

        match combo.pipeline_type:
            case "biological":
                pipelines = pipeline.biological
                constant_options.biological = True
            case "non_biological":
                pipelines = pipeline.non_biological  # type: ignore[assignment]
                constant_options.biological = False
            case _:
                raise NotImplementedError(f"There is no pipeline_type of {combo.pipeline_type}")

        # NOTE: Try this out
        pipelines.collections = collections
        pipelines_instances.append(pipelines)
        # break

    return pipelines_instances
