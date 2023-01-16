import time

from joblib import delayed
from joblib import Parallel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from src.cancerseek.constants import CancerType
from src.cancerseek.constants import ClassificationMode
from src.cancerseek.constants import Classifier
from src.cancerseek.constants import ConstantOptions
from src.cancerseek.constants import FilterMode
from src.cancerseek.constants import KDEMethod
from src.cancerseek.constants import ModelMode
from src.cancerseek.constants import PipelineMode
from src.cancerseek.constants import ValidationMode
from src.cancerseek.hyperparameter_best import hyperparameter_best
from src.cancerseek.paramsearch_util import Parameter
from src.cancerseek.pipeline import combo_pipelines
from src.cancerseek.rfe_t_test import TTest
from src.models.logisitic_regression import LogisticRegression as LogReg
from src.models.multi_layer_perceptron import MultiLayerPerceptron
from src.models.random_forest import RandomForest
from src.models.support_vector_machine import SupportVectorMachine

"""
PIPELINE FRAMEWORK

For each cancer type and classification mode:
    1) RFE - find best feature set
    2) Hyperparameter optimisation - find best parameters for classifiers
    3) No RFE - get files for validation figures
"""

CACHE_PATH = "pipeline_cache"
REFRESH_CACHE = False
PIPELINE_FLAG = PipelineMode.TESTING
RESULTS_PATH = f"Results/{PIPELINE_FLAG.value}"
VALIDATION_FLAG = ValidationMode.NO_RFE
BEST_PARAMS_USED = False

cancer_types = [
    CancerType.BREAST,
    CancerType.COLORECTUM,
    CancerType.ESOPHAGUS,
    CancerType.LIVER,
    CancerType.LUNG,
    CancerType.OVARY,
    CancerType.PANCREAS,
    CancerType.STOMACH,
    CancerType.PANCANCER,
]

classification_modes = [
    ClassificationMode.CANCER_HEALTHY,
    ClassificationMode.CANCER_OTHERCANCERS,
    ClassificationMode.CANCER_REST,
]

classifiers = {
    RandomForest: [Parameter(RandomForestClassifier, *el) for el in Classifier.RF.value],
    LogReg: [Parameter(LogisticRegression, *el) for el in Classifier.LR.value],
    MultiLayerPerceptron: [Parameter(MLPClassifier, *el) for el in Classifier.MLP.value],
    SupportVectorMachine: [Parameter(SVC, *el) for el in Classifier.SVM.value],
}

kde_methods = [KDEMethod.NONE]

model_modes = [
    ModelMode.PROTEINS_ONLY,
    ModelMode.FULL_MODEL,
]

filter_modes = [
    FilterMode.POST_T_TEST,
    FilterMode.EVERY_PROTEIN,
]

# Setup for recursive feature elimination
if PIPELINE_FLAG == PipelineMode.VALIDATING and VALIDATION_FLAG == ValidationMode.RECURSIVE_FEATURE_ELIMINATION:

    classifiers = {
        RandomForest: [
            Parameter(RandomForestClassifier, "n_estimators", [100]),
            Parameter(RandomForestClassifier, "max_depth", [None]),
        ]
    }

    kde_methods = [
        KDEMethod.NONE,
        KDEMethod.IMBALANCED_HEALTHY,
        KDEMethod.IMBALANCED_EARLY,
        KDEMethod.BALANCED,
    ]

    model_modes = [ModelMode.PROTEINS_ONLY]

    filter_modes = [FilterMode.EVERY_PROTEIN]

# Setup for hyperparameter optimisation
elif PIPELINE_FLAG == PipelineMode.VALIDATING and VALIDATION_FLAG == ValidationMode.HYPERPARAMETER_OPTIMISATION:

    classifiers = {
        RandomForest: [Parameter(RandomForestClassifier, *el) for el in Classifier.RF.value],
        LogReg: [Parameter(LogisticRegression, *el) for el in Classifier.LR.value],
        MultiLayerPerceptron: [Parameter(MLPClassifier, *el) for el in Classifier.MLP.value],
        SupportVectorMachine: [Parameter(SVC, *el) for el in Classifier.SVM.value],
    }

    kde_methods = [KDEMethod.NONE]

    model_modes = [ModelMode.PROTEINS_ONLY]

    filter_modes = [FilterMode.POST_T_TEST]

# Setup for final generation of pkl files and figures
elif PIPELINE_FLAG == PipelineMode.VALIDATING and VALIDATION_FLAG == ValidationMode.NO_RFE:

    BEST_PARAMS_USED = True

    classifiers = {
        RandomForest: [Parameter(RandomForestClassifier, *el) for el in Classifier.RF.value],
        LogReg: [Parameter(LogisticRegression, *el) for el in Classifier.LR.value],
        MultiLayerPerceptron: [Parameter(MLPClassifier, *el) for el in Classifier.MLP.value],
        SupportVectorMachine: [Parameter(SVC, *el) for el in Classifier.SVM.value],
    }

    kde_methods = [KDEMethod.NONE]

    model_modes = [
        ModelMode.PROTEINS_ONLY,
        ModelMode.FULL_MODEL,
    ]

    filter_modes = [
        FilterMode.POST_T_TEST,
        FilterMode.EVERY_PROTEIN,
    ]

constant_options = ConstantOptions(
    RESULTS_PATH,
    CACHE_PATH,
    REFRESH_CACHE,
    PIPELINE_FLAG,
    VALIDATION_FLAG,
    BEST_PARAMS_USED,
)

# Parallelised pipeline
start = time.time()
with Parallel(n_jobs=1, verbose=50) as parallel:
    delayed_funcs = [
        delayed(lambda x: x.run_pipeline())(pipe_instance)
        for model, params in classifiers.items()
        for pipe_instance in combo_pipelines(
            [c.value for c in cancer_types],
            classification_modes,
            constant_options,
            kde_methods,
            model_modes,
            filter_modes,
            params=params,
            model=model,
        )
    ]
    value = parallel(delayed_funcs)
print(f"Total time taken: {time.time() - start} seconds.")

if PIPELINE_FLAG == PipelineMode.VALIDATING:
    if VALIDATION_FLAG == ValidationMode.RECURSIVE_FEATURE_ELIMINATION:
        ttest = TTest()
        with Parallel(n_jobs=-1, verbose=50) as parallel:
            delayed_funcs = [delayed(ttest.rfe_t_test)(cancer.value) for cancer in cancer_types]
            parallel(delayed_funcs)
    elif VALIDATION_FLAG == ValidationMode.HYPERPARAMETER_OPTIMISATION:
        for cancer in cancer_types:
            hyperparameter_best(cancer.value)
