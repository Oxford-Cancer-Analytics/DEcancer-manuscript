from __future__ import annotations

from typing import Literal
from typing import NamedTuple
from typing import ParamSpec
from typing import TYPE_CHECKING
from typing import TypeAlias
from typing import TypeVar

import xgboost as xgb
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


if TYPE_CHECKING:
    from .models.logisitic_regression import LogisticRegression as LogReg
    from .models.multi_layer_perceptron import MultiLayerPerceptron
    from .models.random_forest import RandomForest
    from .models.support_vector_machine import SupportVectorMachine
    from .models.decision_tree import DecisionTree
    from .models.xgboost import XGBoost

    ModelEstimator: TypeAlias = (
        LogReg | MultiLayerPerceptron | RandomForest | SupportVectorMachine | DecisionTree | XGBoost
    )

Models = Literal[
    "multi_layer_perceptron",
    "logistic_regression_l2",
    "support_vector_machine",
    "random_forest",
    "decision_tree",
    "xgboost",
]

Preprocessors = Literal[
    "both_fs",
    "multisurf",
    "uni_multi",
    "no_preprocessing",
]

DataTypes = Literal["control", "all"]
KDEType = Literal["balanced", "imbalanced_early", "imbalanced_healthy", "none"]
Proteins: TypeAlias = list[str]

Pipelines = Literal["biological", "non_biological"]
BaseEstimator: TypeAlias = (
    RandomForestClassifier | LogisticRegression | MLPClassifier | SVC | DecisionTreeClassifier | xgb.XGBClassifier
)

KArgs = ParamSpec("KArgs")
T = TypeVar("T", bound=Pipelines)


class Frames(NamedTuple):
    """A set of preprocessed data."""

    name: KDEType
    data: DataFrame
