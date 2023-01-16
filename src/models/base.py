from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Mapping
from typing import Sequence

import numpy as np
from pandas import DataFrame
from pandas import Series
from sklearn.svm import SVC

from .._types import Frames
from .._types import KDEType
from .._types import Preprocessors
from ..constants import Constants
from ..feature_selection import IntersectedFeatureSelector
from ..feature_selection import KDEMultiSurf
from ..kde import KDE
from ..parameters import Parameter


class Preprocess(KDE, KDEMultiSurf, IntersectedFeatureSelector):
    """A combination of all preprocessing classes."""

    ...


class Model(Preprocess, metaclass=ABCMeta):
    """A class for defining the Base Model.

    This Model class inherits from all preprocessing classes and KDE
    with the metaclass of the built-in ABCMeta class. Abstract methods
    are used to enforce consistency for subclasses extending this base
    class. This class should not be directly instantiated.


    Attributes
    ----------
    data
        A dataframe of unprocessed data for a model to be built upon.

    train_info
        Holds the values for each set of Kernel Density estimates.
    """

    __slots__ = ("data", "train_info")

    def __init__(self, data: DataFrame):
        super().__init__()
        self.data = data
        self.collection: str
        self.preprocessor: Preprocessors
        self.train_info: dict[str, Any] = {"feature_importances": np.array([])}

    @abstractmethod
    def preprocess(
        self, train_idx: np.ndarray, validation_idx: np.ndarray, **kwargs: object
    ) -> tuple[Frames, DataFrame, Series]:
        """Abstract preprocess method. Implementation in subclass."""
        ...

    @abstractmethod
    def train(
        self,
        frames: Frames,
        x_valid: DataFrame,
        y_valid: Series,
        index: int,
        **hyperparameters: Mapping[str, Any],
    ) -> None:
        """Abstract train method. Implementation in subclass."""
        ...

    def flush(self, kde_name: KDEType) -> None:
        """Reset of the temporary values for each MCCV iteration.

        All values are saved in another dataframe corresponding to
        their parameters and so this temporary store can be  reused
        with a new set of values.
        """
        self.train_info = {kde_name: {0: []}}

    def _validate_kwargs(self, **kwargs) -> Mapping[str, bool | tuple[int, int]]:  # type: ignore
        """Validate input of **kwargs.

        Parameters
        ----------
        **kwargs
            Extra arguments passed as flags.  Includes `multisurf`,
            `uni_multi`, `both_fs` or `no_preprocessing` as
            preprocessing and `KDEType` as KDE application.

        Returns
        -------
            The validated kwargs, otherwise raises a ValueError.

        Raises
        -------
        ValueError
            Raised if all options are False or there is more than 1
            of `multisurf`, `uni_multi` or `both_fs` set to True.
        """
        options = {
            "uni_multi": False,
            "multisurf": False,
            "both_fs": True,
            "no_preprocessing": False,
        }

        if kwargs.get("multisurf") or kwargs.get("uni_multi") or kwargs.get("no_preprocessing"):
            options["both_fs"] = False
            options |= kwargs  # type: ignore[arg-type]
        else:
            options |= kwargs  # type: ignore[arg-type]

        if all(value is False for value in list(options.values())[: len(options)]):
            raise ValueError(f"One of [{', '.join(options.keys())}] expected as a keyword argument.")

        if sum(list(options.values())[: len(options) - 1]) > 1:
            raise ValueError(
                f"One of {list(options.keys())[:len(options)]} must be True. "
                f"Expected 1, received {sum(list(options.values())[:len(options) - 1])}."
            )

        return options

    def _set_random_state(self, params: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for param in params:
            if "random_state" in param:
                param |= {"random_state": Constants.RANDOM_STATE.value}

        return params

    def _generate_hyperparameters(self, params: Sequence[Parameter]) -> list[dict[str, Any]]:
        def hyperparam_recipe(params: Sequence[Parameter]) -> list[dict[str, Any]]:
            if len(params) == 1:
                if params[0].name == "default":
                    hyperparameters = [params[0].model().get_params()]
                    if issubclass(params[0].model, SVC):
                        hyperparameters[0]["probability"] = True
                    return self._set_random_state(hyperparameters)

                param_values = []
                for value in params[0].values:
                    value = value if not isinstance(value, float) else None if np.isnan(value) else value
                    param_values.append({params[0].name: value})
                return param_values
            else:
                other_params = hyperparam_recipe(params[1:])
                res = []
                for value in params[0].values:
                    value = value if not isinstance(value, float) else None if np.isnan(value) else value
                    res.extend([{params[0].name: value, **dd.copy()} for dd in other_params])
                return res

        return hyperparam_recipe(params)
