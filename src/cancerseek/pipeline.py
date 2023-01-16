from __future__ import annotations

import ast
import os
import pickle
from itertools import product
from pathlib import Path
from typing import cast
from typing import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas import DataFrame
from src.parameters import Parameter

from .constants import ClassificationMode
from .constants import ComboParams
from .constants import ConstantOptions
from .constants import FilterMode
from .constants import KDEMethod
from .constants import ModelMode
from .constants import PipelineMode
from .constants import ValidationMode
from .paramsearch_util import ParameterSearch
from .utility import Data
from .utility import DataFeatures

if TYPE_CHECKING:
    from .._types import ModelEstimator


def combo_pipelines(
    cancer: Sequence[str],
    classification_modes: Sequence[ClassificationMode],
    constant_options: ConstantOptions,
    kde_methods: Sequence[KDEMethod] = [KDEMethod.NONE],
    model_modes: Sequence[ModelMode] = [ModelMode.PROTEINS_ONLY],
    filter_modes: Sequence[FilterMode] = [FilterMode.POST_T_TEST],
    *,
    params: list[Parameter],
    model: type[ModelEstimator],
) -> list[Pipeline]:
    """Produces the pipeline combinations based on the inputs.

    Parameters
    ----------
    cancer
        The types of cancer.
    classification_modes
        A combination of classification modes.
    params
        A list of Parameter's.
    model
        The type of custom model .
    cancer
        The type of cancer.
    kde_methods, optional
        A combination of augmentation methods,
        by default [KDEMethod.NONE]
    model_modes, optional
        A combination of model modes, by default [ModelMode.PROTEINS_ONLY]
    filter_modes, optional
        A combination of filter modes, by default [FilterMode.POST_T_TEST]

    Returns
    -------
        A sequence of pipeline combinations.
    """
    pipelines_instances = []
    for combo in product(cancer, classification_modes, kde_methods, model_modes, filter_modes):
        combo_options = ComboParams(*combo)
        try:
            pipelines_instances.append(Pipeline(constant_options, combo_options, params, model))
        except FileNotFoundError:
            print(f"Skipping Results/validation/{combo_options.collection}/best_hyperparameters")
            continue
        except ValueError:
            # Can only be raised during Testing
            continue

    return pipelines_instances


class Pipeline:
    """The main pipeline class for cancerseek."""

    def __init__(
        self,
        constant_options: ConstantOptions,
        combos: ComboParams,
        params: Sequence[Parameter],
        model: type[ModelEstimator],
    ) -> None:

        self.constant_options = constant_options
        self.combos = combos
        self.hyperparameters = params
        self.model = model
        self.cancer = combos.collection
        self.data = Data()
        self.base_path = f"Results/{self.constant_options.pipeline_flag.value}"

        if self.constant_options.pipeline_flag == PipelineMode.VALIDATING:
            self.validation = Validation(constant_options, combos)

        if self.constant_options.best_params:
            self.hyperparameters = self._get_best_parameters()

    def run_pipeline(self) -> None:
        """Runs the pipeline based on the options for each combination."""
        sample_ids, y_data, y_source, y_stage, x_ordinal = self._select_data(
            self.cancer,
            self.combos.classification_mode,
            *self._categorise_data(),
        )

        print("After selecting data, initialise Classifier.")
        print(f"Model: {self.model.__name__}")  # type: ignore

        search = ParameterSearch(
            self.model,
            self.hyperparameters,
            self.constant_options.pipeline_flag,
            self.combos,
            preprocessing_params={"kde": self.combos.kde_method},
        )

        x_df = np.copy(x_ordinal[:, :-4]) if self.combos.model_mode == ModelMode.PROTEINS_ONLY else np.copy(x_ordinal)
        data_bundle = sample_ids, x_df, y_data, y_source, y_stage

        if self.constant_options.pipeline_flag == PipelineMode.TESTING:
            self.test_pipeline(search, data_bundle)
        elif self.constant_options.pipeline_flag == PipelineMode.VALIDATING:
            self.validation_pipeline(search, data_bundle)

    # Import union features file
    def post_t_test(self, df: np.ndarray) -> np.ndarray:
        """Reads in the set of union features.

        Parameters
        ----------
        df
            The original set of data features.

        Returns
        -------
            Filtered features.
        """
        union_path = (
            f"{self.base_path}/{self.cancer}/post_t_test/"
            f"{self.combos.classification_mode.value}_{self.combos.model_mode.value}_smallestSelected.xlsx"
        )
        union_df = pd.read_excel(
            union_path,
            sheet_name="union_features",
            header=0,
            index_col=0,
            engine="openpyxl",
        )
        union_list = [protein for protein in union_df[0]]
        proteins_mask = [
            self.data.features.features.index(el) for el in self.data.features.features if el not in union_list
        ]
        proteins_mask.sort()

        df_filtered = np.delete(df, proteins_mask, 1)

        return df_filtered

    def test_pipeline(
        self,
        search: ParameterSearch,
        data_bundle: tuple[np.ndarray, ...],
    ) -> None:
        """Runs the testing pipeline.

        Parameters
        ----------
        search
            The instance of a ParameterSearch.
        data_bundle
            A tuple of the original data.
        """
        print("THIS IS THE TESTING PIPELINE")

        sample_ids, x_df, y_data, y_source, y_stage = data_bundle
        try:
            x_df_post_t_test = self.post_t_test(x_df)
            testing_combos = zip([x_df, x_df_post_t_test], ["all", "postTTest"])
        except FileNotFoundError:
            # Only the fullModel_all files are in
            # {self.base_path}/post_t_test/
            testing_combos = zip([x_df], ["all"])

        for df, signature in testing_combos:

            search.init_data(sample_ids, df, y_data, y_source, y_stage)
            results = search.run_search()

            savedir = (
                f"{self.constant_options.results_path}/{self.cancer}/"
                f"{self.combos.classification_mode.value}/{self.combos.model_mode.value}_{signature}"
            )
            Path(savedir).mkdir(parents=True, exist_ok=True)
            savepath = f"{savedir}/{search.model.__name__}Test"  # type: ignore
            print(f"Saving to {savepath}.pkl ...")

            with open(f"{savepath}.pkl", "wb") as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pd.DataFrame(results).to_excel(f"{savepath}.xlsx")

    def validation_pipeline(
        self,
        search: ParameterSearch,
        data_bundle: tuple[np.ndarray, ...],
    ) -> None:
        """Runs the validation pipeline.

        Parameters
        ----------
        search
            The instance of a ParameterSearch.
        data_bundle
            A tuple of the original data.
        """
        print("THIS IS THE VALIDATION PIELINE")

        sample_ids, x_df, y_data, y_source, y_stage = data_bundle

        if self.validation.validation_flag == ValidationMode.RECURSIVE_FEATURE_ELIMINATION:
            rfe_steps = self.validation.recursive_feature_elimination(
                data_bundle, search, self.combos.model_mode, self.data.features
            )

            savepath = f"{self.validation.result_path}/RFE_{self.validation.kde_suffix}_{self.combos.model_mode.value}"
            with open(f"{savepath}.pkl", "wb") as handle:
                pickle.dump(rfe_steps, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pd.DataFrame(rfe_steps).T.to_excel(f"{savepath}.xlsx")
        elif self.validation.validation_flag == ValidationMode.HYPERPARAMETER_OPTIMISATION:

            assert self.combos.kde_method == KDEMethod.NONE
            assert self.combos.model_mode == ModelMode.PROTEINS_ONLY
            assert self.combos.filter_mode == FilterMode.POST_T_TEST

            try:
                x_df = self.post_t_test(x_df)
            except FileNotFoundError:
                # Only the fullModel_all files are in
                # {self.base_path}/post_t_test/
                return

            search.init_data(sample_ids, x_df, y_data, y_source, y_stage)
            results = search.run_search()

            savepath = f"{self.validation.result_path}/{search.model.__name__}Validation"  # type: ignore
            print(f"Saving to {savepath}.pkl ...")

            with open(f"{savepath}.pkl", "wb") as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pd.DataFrame(results).T.to_excel(f"{savepath}.xlsx")
        elif self.validation.validation_flag == ValidationMode.NO_RFE:
            rfe_path = (
                f"{self.validation.result_path}/feature_elimination/"
                f"RFE_{self.validation.kde_suffix}_{self.combos.model_mode.value}.pkl"
            )

            if self.combos.filter_mode == FilterMode.FILTER_BY_RFE and Path(rfe_path).is_file():
                with open(rfe_path, "rb") as handle:
                    rfe_dict = pickle.load(handle)
                rfe_df = pd.DataFrame(rfe_dict).T

                # Identify subset of proteins yielding best AUC
                rfe_idx = rfe_df["auc"].astype("float").idxmax()
                if isinstance(rfe_idx, int):
                    removed_proteins = [i for i in rfe_df.loc[: rfe_idx + 1, "protein_removed"]]

                    # Select only these proteins from numpy array
                    proteins_mask = [self.data.features.features.index(el) for el in removed_proteins]
                    proteins_mask.sort()
                    x_df = np.delete(x_df, proteins_mask, 1)
            elif self.combos.filter_mode == FilterMode.POST_T_TEST:
                try:
                    x_df = self.post_t_test(x_df)
                except FileNotFoundError:
                    # Only the fullModel_all files are in
                    # {self.base_path}/post_t_test/
                    return

            search.init_data(sample_ids, x_df, y_data, y_source, y_stage)
            results = search.run_search()

            savepath = f"{self.validation.result_path}/{search.model.__name__}Validation"  # type: ignore
            os.makedirs(savepath, exist_ok=True)
            print(f"Saving to {savepath}.pkl ...")

            with open(f"{savepath}.pkl", "wb") as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pd.DataFrame(results).T.to_excel(f"{savepath}.xlsx")

    def _categorise_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Preprocesses the data and features.

        Returns
        -------
            Categorised sample ids, original xy data, tumor and stage data.
        """
        sample_ids, x_combined_ordinal = self.data.data_preprocess(
            self.constant_options.cache_path, self.constant_options.refresh_cache
        )
        y_tumor, y_stage, tumor_categories, _ = self.data.feature_preprocess(
            self.constant_options.cache_path, self.constant_options.refresh_cache
        )

        xy_combined_ordinal = np.concatenate([x_combined_ordinal, y_tumor], axis=1)

        return sample_ids, tumor_categories, y_stage, xy_combined_ordinal

    def _get_best_parameters(self) -> Sequence[Parameter]:
        """Loads and finds the best hyperparameters.

        This is used when the BEST_PARAMS_USED is True to
        generate the best set of hyperparameters based on
        the combinations for each pipeline.

        Returns
        -------
            A sequence of `Parameter`'s for the specific model.

        Raises
        ------
        ValueError
            If the model is not the same as the model when
            searching for the best hyperparameters.
        """
        best_params = pd.read_excel(
            f"{self.base_path}/{self.cancer}/best_hyperparameters/best_hyperparameters.xlsx",
            index_col=0,
            engine="openpyxl",
        )
        model_name = str(self.model).split(".")[-1].split("'")[0]
        model = next(iter(self.hyperparameters)).model

        if self.constant_options.pipeline_flag == PipelineMode.VALIDATING:
            best_model_row = best_params[
                (best_params["classification_mode"] == self.combos.classification_mode.value)
                & (best_params["classifier"] == f"{model_name}Validation")
            ]

        else:
            class_mode_rows = cast(
                DataFrame, best_params[best_params["classification_mode"] == self.combos.classification_mode.value]
            )
            best_auc_idx = cast(int, class_mode_rows["auc"].idxmax())
            best_model_row = class_mode_rows.loc[[best_auc_idx], :]
            if best_model_row["classifier"].values[0] != f"{model_name}Validation":
                raise ValueError(
                    f"Different classifier: {best_model_row['classifier'].values[0]} is not {model_name}Validation."
                )

        (params,) = best_model_row["params"].values
        params = {k: [v] for k, v in ast.literal_eval(params).items()}
        return [Parameter(model, *el) for el in tuple(zip(params, params.values()))]

    def _select_data(
        self,
        cancer: str,
        classification_mode: ClassificationMode,
        sample_ids: np.ndarray,
        tumor_categories: np.ndarray,
        y_stage: np.ndarray,
        xy_combined_ordinal: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        """Selects the specific data based on ClassificationMode.

        Parameters
        ----------
        cancer
            The type of cancer.
        classification_mode
            The type of ClassificationMode.
        sample_ids
            The original set of `sample_ids`.
        tumor_categories
            The original set of `tumor_categories`
        y_stage
            The y values for cancer stage.
        xy_combined_ordinal
            The original x and y data.

        Returns
        -------
            A tuple of sample ids with x and y data.
        """
        # Separating x,y_data,sample IDs, data
        # from the variable xy_combined.
        x_ordinal = xy_combined_ordinal[:, :-1]
        y_ = xy_combined_ordinal[:, -1]

        cancer_idx = (
            np.where(tumor_categories != "Normal")[0]
            if cancer == "pancancer"
            else np.where(tumor_categories == cancer)[0]
        )
        healthy_idx = np.where(tumor_categories == "Normal")[0]
        y_source = np.copy(y_)

        # 'y_stage' is where we keep the cancer
        # stage data. Reshaping to 1D array.
        y_stage = y_stage.reshape(-1)

        print(f"Cancer type is {cancer}.")

        if classification_mode == ClassificationMode.THREE_CLASSES:
            # y_data Encoding: healthy = 0,
            # other cancers = 1, cancer = 2
            print(f"Classification type is set to {classification_mode.value}.")

            y_data = np.ones(y_.shape[0])
            y_data[np.in1d(y_, healthy_idx)] = 0
            y_data[np.in1d(y_, cancer_idx)] = 2
        elif classification_mode == ClassificationMode.CANCER_HEALTHY:
            # y_data Encoding: healthy = 0,
            # cancer = 1
            print(f"Classification type is set to {classification_mode.value}.")

            y_data = np.ones(y_.shape[0])
            y_data = y_data * -1

            y_data[np.in1d(y_, healthy_idx)] = 0
            y_data[np.in1d(y_, cancer_idx)] = 1

            mask = y_data != -1

            sample_ids = sample_ids[mask]
            y_data = y_data[mask]
            x_ordinal = x_ordinal[mask, :]
            y_source = y_source[mask]
            y_stage = y_stage[mask]
        elif classification_mode == ClassificationMode.CANCER_OTHERCANCERS:
            # y_data Encoding: other
            # cancers = 0, cancer = 1
            print(f"Classification type is set to {classification_mode.value}.")

            y_data = np.zeros(y_.shape[0])

            y_data[np.in1d(y_, healthy_idx)] = -1
            y_data[np.in1d(y_, cancer_idx)] = 1

            mask = y_data != -1

            sample_ids = sample_ids[mask]
            y_data = y_data[mask]
            x_ordinal = x_ordinal[mask, :]
            y_source = y_source[mask]
            y_stage = y_stage[mask]
        else:
            # y_data Encoding: rest = 0,
            # cancer = 1
            print(f"Classification type is set to {classification_mode.value}.")

            y_data = np.zeros(y_.shape[0])
            y_data[np.in1d(y_, cancer_idx)] = 1

        return sample_ids, y_data, y_source, y_stage, x_ordinal


class Validation:
    """Validation specific methods."""

    def __init__(
        self,
        constant_options: ConstantOptions,
        combos: ComboParams,
    ) -> None:
        self.constant_options = constant_options
        self.combos = combos
        self.cancer = combos.collection

        self.validation_flag = constant_options.validation_flag
        self.kde_method = combos.kde_method

        self._create_dirs()

    def recursive_feature_elimination(
        self,
        data_bundle: tuple[np.ndarray, ...],
        search: ParameterSearch,
        model_type: ModelMode,
        data_features: DataFeatures,
    ) -> dict[int, dict]:
        ...

    def _create_dirs(self) -> None:
        """Creates the relevant validation result directories."""
        kde_suffix = (
            str(self.kde_method.value)
            if self.kde_method == KDEMethod.NONE
            else f'{self.kde_method.value[0]}_{"_".join([str(k) for k in self.kde_method.value[1]])}'
        )
        self.kde_suffix = kde_suffix
        base_path = f"{self.constant_options.results_path}/{self.cancer}/{self.combos.classification_mode.value}"

        if self.constant_options.validation_flag == ValidationMode.RECURSIVE_FEATURE_ELIMINATION:
            result_path = f"{base_path}/{kde_suffix}/{self.validation_flag.value}"
        elif self.constant_options.validation_flag == ValidationMode.HYPERPARAMETER_OPTIMISATION:
            result_path = f"{base_path}/{self.validation_flag.value}"
        else:
            result_path = f"{base_path}/{kde_suffix}/{self.combos.model_mode.value}_{self.combos.filter_mode.value}"

        Path(result_path).mkdir(parents=True, exist_ok=True)
        self.result_path = result_path
