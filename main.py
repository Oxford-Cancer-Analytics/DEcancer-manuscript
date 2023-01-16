from __future__ import annotations

import os
from typing import Any
from typing import cast
from typing import TYPE_CHECKING

from joblib import dump
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from src.constants import AugmentationMethod
from src.constants import PipelineStage
from src.data_models import ConstantOptions
from src.parameters import Parameter
from src.parameters import rf_params
from src.pipeline import combo_pipelines

if TYPE_CHECKING:
    from src.pipeline.biological import Biological
    from src.pipeline.non_biological import NonBiological


constant_options = ConstantOptions(stage=PipelineStage.VALIDATION, feature_selection=True)

pipes: list[Biological | NonBiological] = []
params = classifiers = [
    [Parameter(RandomForestClassifier, "default")],
]

for param in params:
    pipes.extend(
        combo_pipelines(
            ["DP"],
            [AugmentationMethod.NONE],
            ["multisurf"],
            ["non_biological"],
            ["all"],
            hyperparameters=param,
            constant_options=constant_options,
        )
    )


if __name__ == "__main__":
    for pipe in pipes:
        if TYPE_CHECKING:
            pipe = cast(NonBiological, pipe)

        file_start = f"Results/{pipe.combo_options.pipeline_type}/{pipe.collection}/{pipe.model_name}"
        file_path = f"{file_start}/Final_proteins_sets_NB_{pipe.model_name}.joblib"
        os.makedirs(file_start, exist_ok=True)

        pipe.data = cast(list[dict[str, Any]], pipe.run_pipeline(file_path))
        # Data is used to select features for rest of pipeline

        pipe.combo_options.preprocessor = "no_preprocessing"
        pipe.hyperparameters = [Parameter(RandomForestClassifier, *el) for el in rf_params]
        pipe.constant_options.feature_selection = False

        # Hyperparameter tuning
        pipe.run_pipeline(file_path, optimise_params=True)
        model = next(iter(pipe.hyperparameters)).model
        param_values, param_data = pipe.get_best_hyperparameters(pipe.kde_method, version=1)[pipe.collection][
            pipe.model_name
        ]
        param_names = param_data.index.names
        params = [Parameter(model, name, [value]) for name, value in zip(param_names, param_values)]

        with open(file_path, "rb") as handle:
            proteins = load(handle)

        # Recursive Feature Elimination
        rfe_data = pipe._run_rfe_loop(pipe.collection, pipe.kde_method, params, proteins=proteins)
        final_joblib_dir = (
            f"Results/{pipe.combo_options.pipeline_type}/{pipe.collection}/"
            f"{pipe.kde_method}/v{pipe.constant_options.version}/"
        )

        final_joblib_path = final_joblib_dir + f"{pipe.collection}_Final_results_{pipe.model_name}_best_params.joblib"
        os.makedirs(final_joblib_dir, exist_ok=True)
        with open(final_joblib_path, "wb") as file:
            dump(rfe_data, file)

        excel_path = (
            f"Results/{pipe.combo_options.pipeline_type}/{pipe.collection}/{pipe.kde_method}/"
            f"{pipe.collection}_{pipe.model_name}_best_params.xlsx"
        )

        full_protein_sets = pipe.best_protein_set_max(excel_path, final_joblib_path)
        stats_results = pipe.statistical_indicies(full_protein_sets, final_joblib_path, biological=False)
        _ = pipe.max_auc_to_excel(full_protein_sets, excel_path, pipe.collection, stats_results, biological=False)
