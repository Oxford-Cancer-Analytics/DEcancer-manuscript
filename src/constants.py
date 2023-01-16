from enum import Enum


class Constants(Enum):
    """Constant variables."""

    SPLITS = 200  # MCCV splits
    TEST_SIZE = 0.2  # Training
    TRAIN_SIZE = 0.8  # Training
    RANDOM_STATE = 0


class AugmentationMethod(Enum):
    """The different types of kernel density augmentation."""

    NONE = (0, 0)
    BALANCED = (1, 1)
    IMBALANCED_HEALTHY = (1, 4)
    IMBALANCED_EARLY = (4, 1)


class PipelineStage(Enum):
    """Each stage of the pipeline."""

    VALIDATION = "validation"
    TESTING = "testing"
    RECURSIVE_FEATURE_ELIMINATION = "rfe"
