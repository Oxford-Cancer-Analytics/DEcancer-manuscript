from dataclasses import dataclass
from enum import auto
from enum import Enum

from src._types import Preprocessors

mlp_params = ((), (), ...)

lr_params = ((), (), ...)

svm_params = ((), (), ...)

rf_params = ((), (), ...)


class ClassifierType(Enum):
    """The type of classifier."""

    CLASS_2 = 1
    CLASS_3 = 2


class CrossValidation(Enum):
    """The type of cross-validation."""

    K_FOLD = "kf_cv"
    MONTE_CARLO = "mc_cv"


class PipelineMode(Enum):
    """The pipeline mode to use."""

    TESTING = "test"
    VALIDATING = "validation"


class ValidationMode(Enum):
    """The type of validation mode to use.

    Whether to run the validation pipeline with recursive
    feature elimination, optimize hyperparameters or
    without any specific use.
    """

    RECURSIVE_FEATURE_ELIMINATION = "feature_elimination"
    NO_RFE = auto()
    HYPERPARAMETER_OPTIMISATION = "hyper_optimisation"


class CancerType(Enum):
    """The type of cancer."""

    BREAST = "Breast"
    COLORECTUM = "Colorectum"
    ESOPHAGUS = "Esophagus"
    LIVER = "Liver"
    LUNG = "Lung"
    OVARY = "Ovary"
    PANCANCER = "pancancer"
    PANCREAS = "Pancreas"
    STOMACH = "Stomach"


class Classifier(Enum):
    """The type of classifier to use for the respective hyperparameters."""

    RF = rf_params
    MLP = mlp_params
    LR = lr_params
    SVM = svm_params


class ClassificationMode(Enum):
    """The classification mode to use."""

    CANCER_HEALTHY = "cancer-healthy"
    CANCER_OTHERCANCERS = "cancer-otherCancers"
    CANCER_REST = "cancer-rest"
    THREE_CLASSES = "three-classes"


class KDEMethod(Enum):
    """The Kernel Density Method to use."""

    NONE = 0
    IMBALANCED_HEALTHY = (5, (4, 1))
    BALANCED = (5, (1, 1))
    IMBALANCED_EARLY = (5, (1, 4))


class ModelMode(Enum):
    """The model mode to use.

    The full model includes all proteins, DNA, sex, age and ethnicity.
    """

    FULL_MODEL = "fullModel"
    PROTEINS_ONLY = "proteinsOnly"


class FilterMode(Enum):
    """The filter mode to use.

    Filtering the proteins by what is chosen as the best
    during recursive feature elimination, selected via
    t-test, or to use all proteins.
    """

    FILTER_BY_RFE = "bestRfe"
    POST_T_TEST = "postTTest"
    EVERY_PROTEIN = "all"


class FigureFlag(Enum):
    """The type of figures to generate."""

    TEST = auto()
    VALIDATION = auto()


VALIDATION_METHOD = CrossValidation.MONTE_CARLO
TEST_SPLIT = 0.2
SEED = 42


@dataclass
class ConstantOptions:
    """The options for constant data."""

    results_path: str
    cache_path: str
    refresh_cache: bool
    pipeline_flag: PipelineMode
    validation_flag: ValidationMode
    best_params: bool = False


@dataclass
class ComboParams:
    """The options for each combination of parameters."""

    collection: str
    classification_mode: ClassificationMode
    kde_method: KDEMethod
    model_mode: ModelMode
    filter_mode: FilterMode
    preprocessor: Preprocessors = "no_preprocessing"
