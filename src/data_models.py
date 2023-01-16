from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple
from typing import Sequence
from typing import TYPE_CHECKING

from pandas import DataFrame

from ._types import DataTypes
from ._types import Pipelines
from ._types import Preprocessors
from .constants import AugmentationMethod
from .constants import Constants
from .constants import PipelineStage

if TYPE_CHECKING:
    from .pipeline.non_biological import NonBiological
    from .pipeline.biological import Biological


class Data(NamedTuple):
    """A set of unprocessed data, split into training and testing."""

    training: DataFrame = DataFrame()
    testing: DataFrame = DataFrame()
    full: DataFrame = DataFrame()


@dataclass
class ConstantOptions:
    """The options for constant data."""

    version: int = 1
    constants: type[Constants] = Constants
    biological: bool = False
    control_data: bool = False
    stage: PipelineStage = PipelineStage.VALIDATION
    feature_selection: bool = False


@dataclass
class ComboOptions:
    """The options for each combination of parameters."""

    collection: str
    kde_method: AugmentationMethod
    preprocessor: Preprocessors
    pipeline_type: Pipelines
    data_type: DataTypes
    data: Data = Data()


@dataclass
class FigureOptions:
    """The set of options for producing figures.

    `final` is for generating the testing figures and `repeat` is to
    generate the roc_curve data for averaging.
    """

    pipe: NonBiological | Biological
    final: bool = False
    repeat: bool = True


@dataclass
class Ranking:
    """The data used for ranking biologial clusters."""

    order: Sequence = ()
    name: str = "extra_all_alt"
