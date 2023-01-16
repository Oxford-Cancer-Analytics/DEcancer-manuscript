from __future__ import annotations

import os
from typing import Any
from typing import cast
from typing import Sequence

import numpy as np
from joblib import dump
from joblib import load
from pandas import ExcelWriter

from .._types import KDEType
from .._types import Models
from .._types import Pipelines
from ..data_models import ComboOptions
from ..data_models import ConstantOptions
from ..parameters import Parameter
from .shared import SharedPipeline


class NonBiological(SharedPipeline):
    """The non-biological pipeline."""

    def __init__(
        self,
        model_name: Models,
        combo_options: ComboOptions,
        constant_options: ConstantOptions,
        hyperparameters: Sequence[Parameter],
        data: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.combo_options = combo_options
        self.constant_options = constant_options
        self.hyperparameters = hyperparameters
        self.data = data

        # Shortcuts
        self.pipeline: Pipelines = combo_options.pipeline_type
        self.collection = self.combo_options.collection
        self.kde_method = cast(KDEType, self.combo_options.kde_method.name.lower())
        self.preprocessor = combo_options.preprocessor

        self.base_save_path = f"Results/{self.pipeline}/{self.collection}/{self.kde_method}/{self.preprocessor}"
        os.makedirs(self.base_save_path, exist_ok=True)

    @property  # type: ignore[misc]
    def raw_data(self) -> list[dict]:
        """The raw data which holds all collections.

        Returns
        -------
            Data of each collection across 200 MCCV folds.
        """
        if self.data is None:
            with open(
                f"Results/ROC_AUC/{self.model_name}/{self.preprocessor}/{self.kde_method}/{self.model_name}.joblib",
                "rb",
            ) as f:
                return load(f)
        else:
            return self.data

    def meta_averages(self, results: dict[str, dict]) -> dict[str, dict[str, np.floating]]:
        """Retrieves the average AUC score.

        The data is across all 200 MCCV folds for each
        collection and augmentation method.

        Parameters
        ----------
        results
            The full dictionary of auc scores

        Returns
        -------
            A compact dictionary mapping with averaged auc scores.
        """
        meta_collection: dict[str, dict[str, np.floating]] = {}

        for collection, result_data in results.items():
            meta_collection[collection] = {}
            for kde, folds in result_data.items():
                if kde != self.kde_method:
                    continue
                avg = []
                for _, aucs in folds.items():
                    avg.extend(aucs["auc"])

                meta_collection[collection][kde] = np.mean(avg)

        with open(f"{self.base_save_path}/meta.joblib", "wb") as file:
            dump(meta_collection, file)

        return meta_collection

    def best_protein_set_max(self, path: str, results_path: str) -> dict[str, dict]:
        """Calculates the best set of proteins to use.

        All data is exported to an Excel sheet which includes
        all of the collections, augmentation methods and the calculated
        auc values.

        Parameters
        ----------
        path
            The excel path to save data to.
        results_path
            The RFE results path.

        Returns
        -------
            A comprehensive mapping of proteins, recursive
            elimination values and other metadata, keyed by
            collection and augmentation method.
        """
        full_protein_sets: dict[str, dict] = {}
        for collection in self.collections:
            full_protein_sets[collection] = {}
            with ExcelWriter(path) as writer:  # type: ignore[abstract]
                for kde in self.kde_methods:
                    print(collection, kde)
                    try:
                        with open(results_path, "rb") as f:
                            final_non_bio = load(f)
                    except FileNotFoundError:
                        continue

                    if kde not in results_path:
                        continue

                    rfe = {}
                    non_bio_collection = final_non_bio[collection]
                    for num, kde_results in non_bio_collection.items():
                        aucs = []
                        protein_removed = kde_results[kde]["protein_removed"]
                        aucs = kde_results[kde]["aucs"]

                        sem = np.std(aucs, ddof=1) / np.sqrt(np.size(aucs))
                        avg = np.mean(aucs), np.std(aucs), sem, protein_removed
                        rfe[num] = avg

                    (rfe_removal, best_variable_set,) = self._create_best_protein_dataframe(
                        # mypy can't work out the *results type. PEP 646
                        rfe,  # type: ignore[arg-type]
                        writer,
                        kde,
                    )

                    smallest_within_std = min(
                        {k: v for k, v in rfe.items() if k not in rfe_removal},
                    )

                    proteins = self._find_proteins_within_std(rfe, best_variable_set)  # type: ignore[arg-type]

                    full_protein_sets[collection][kde] = (
                        proteins,
                        rfe_removal,
                        rfe,
                        smallest_within_std,
                    )
        return full_protein_sets
