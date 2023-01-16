from __future__ import annotations

import os
import pprint
import shutil
import time
from typing import cast
from typing import Iterable
from typing import Mapping
from typing import Sequence
from typing import TYPE_CHECKING

import numpy as np
from joblib import dump
from joblib import load
from pandas import DataFrame
from pandas import ExcelWriter

from .._types import KDEType
from .._types import Models
from .._types import Pipelines
from .._types import Proteins
from ..parameters import Parameter
from .kegg import KEGG
from .shared import SharedPipeline

if TYPE_CHECKING:
    from ..data_models import ComboOptions
    from ..data_models import ConstantOptions


pp = pprint.PrettyPrinter(indent=2, width=200, compact=True, sort_dicts=False)


class Biological(SharedPipeline):
    """The biological pipeline."""

    NUM_OF_CLASSES = 16

    def __init__(
        self,
        model_name: Models,
        combo_options: ComboOptions,
        constant_options: ConstantOptions,
        hyperparameters: Sequence[Parameter],
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.combo_options = combo_options
        self.constant_options = constant_options
        self.hyperparameters = hyperparameters

        # Shortcuts
        self.collection = self.combo_options.collection
        self.kde_method = cast(KDEType, self.combo_options.kde_method.name.lower())
        self.preprocessor = combo_options.preprocessor
        self.pipeline: Pipelines = combo_options.pipeline_type

        self.kegg = KEGG(self.collections, self.kde_methods)

        self.base_save_path = f"Results/{self.pipeline}/{self.collection}/{self.kde_method}/{self.preprocessor}"
        os.makedirs(self.base_save_path, exist_ok=True)

    def run_kegg(self, **kwargs: Mapping[str, DataFrame]) -> None:
        """Runs the KEGG pathway pipeline.

        Parameters
        ----------
        **kwargs
            Options that are forwarded to `KEGG.clusters()`
        """
        self._kegg_clusters(self.combo_options.collection, self.combo_options.data.training)
        self.kegg.split_classes(**kwargs)  # type: ignore[arg-type]

    def best_protein_set_max(self, path: str, results_path: str) -> dict[str, dict]:  # noqa: U100
        """Gets the best set of proteins for each biological cluster.

        Parameters
        ----------
        path
            The excel path to save data to.
        results_path
            The RFE results path.

        Returns
        -------
            A data model of best proteins per biological cluster,
            including recursive feature elimination.
        """
        full_protein_sets: dict[str, dict] = {}
        for collection in self.collections:
            full_protein_sets[collection] = {}
            with ExcelWriter(path) as writer:  # type: ignore[abstract]
                for class_num in range(1, 17):
                    full_protein_sets[collection][class_num] = {}
                    for kde in self.kde_methods:
                        with open(
                            f"Results/biological/{collection}/{kde}/rfe/{class_num}.joblib",
                            "rb",
                        ) as f:
                            biological_clusters = load(f)

                        rfe = biological_clusters[collection]
                        for num, folds in rfe.items():
                            aucs = []
                            proteins = folds["proteins"]
                            for fold, auc in folds.items():
                                if fold == "proteins":
                                    continue
                                aucs.extend(auc["aucs"])

                            sem = np.std(aucs, ddof=1) / np.sqrt(np.size(aucs))
                            avg = np.mean(aucs), np.std(aucs), sem, proteins
                            rfe[num] = avg

                        (
                            rfe_removal,
                            best_variable_set,
                        ) = self._create_best_protein_dataframe(rfe, writer, f"{class_num}_{kde}")

                        smallest_within_std = min({k: v for k, v in rfe.items() if k not in rfe_removal})

                        proteins = self._find_proteins_within_std(rfe, best_variable_set)

                        full_protein_sets[collection][class_num][kde] = (
                            proteins,
                            rfe_removal,
                            rfe,
                            smallest_within_std,
                        )
        return full_protein_sets

    def run_clusters(self, collection: str, partition: DataFrame, ranking: str = "extra_all") -> None:
        """Creates a data file for each biological cluster.

        The biological cluster order is based on the ranking.

        Parameters
        ----------
        collection
            The name of the nanoparticle.
        partition
            The training data.
        ranking, optional
            The type of ranking used for biological clusters,
            by default "extra_all"
        """
        np.random.seed(self.constant_options.constants.RANDOM_STATE.value)
        class_path = f"{self.kegg.result_path}/{collection}/Classes"

        start = time.time()
        for class_num in range(1, self.NUM_OF_CLASSES + 1):
            os.makedirs(
                f"{self.kegg.result_path}/{collection}/{self.kde_method}/{class_num}/",
                exist_ok=True,
            )

            with open(f"{class_path}/{ranking}/{class_num}.joblib", "rb") as e:
                extra = load(e)

            for cluster_num, extra_data in extra.items():
                proteins = extra_data[0]

                self._run_inner_loop(collection, partition, class_num, cluster_num, proteins)

        print("Run time: ", time.time() - start, " seconds.")

    def recursive_feature_elimination(self, collection: str) -> None:
        """Performs recursive protein elimination.

        The proteins are unique and are capped at 80 to match that of the
        validation sample size. Therefore, some  of the classification
        results will be the same due to the ranking of clusters and how
        they are ordered to maintain a protein set of no more than 80. The
        cluster ranking is ordered based on the 200-averaged AUC scores.

        Parameters
        ----------
        ollection: str
            The nanoparticle to be used.
        """
        np.random.seed(self.constant_options.constants.RANDOM_STATE.value)
        start = time.time()

        class_path = f"Results/{self.pipeline}/{collection}/{self.kde_method}/Classes"
        rfe_path = f"Results/{self.pipeline}/{collection}/{self.kde_method}/rfe"

        completed: dict[int, Iterable] = {}
        for class_ in range(1, self.NUM_OF_CLASSES + 1):
            os.makedirs(f"{rfe_path}/{class_}", exist_ok=True)

            with open(f"{class_path}/{class_}.joblib", "rb") as e:
                class_info = load(e)

                # Clusters sorted based on AUC
                cluster_order = sorted(class_info, key=lambda x: class_info[x]["kde"][0], reverse=True)

            rfe_proteins, protein_cluster = self._find_rfe_proteins(cluster_order, class_info)

            if protein_cluster in completed.values():
                class_num = list(completed.keys())[list(completed.values()).index(protein_cluster)]

                # Copy the previous class to the next class and continue
                # to the next
                shutil.copyfile(
                    f"{rfe_path}/{class_num}.joblib",
                    f"{rfe_path}/{class_}.joblib",
                )
                continue

            completed[class_] = protein_cluster
            print(collection, "Class", class_)

            rfe_results = self._run_rfe_loop(collection, self.kde_method, self.hyperparameters, proteins=rfe_proteins)

            with open(f"{rfe_path}/{class_}.joblib", "wb") as f:
                dump(rfe_results, f)

        print("Run time: ", time.time() - start, " seconds.")

    def _find_rfe_proteins(self, order: list[str], info: dict) -> tuple[Proteins, Proteins]:
        """Find all the unique proteins which are no more than 80.

        Parameters
        ----------
        order
            The order of the current configuration.
        info
            The data for the current configuration.

        Returns
        -------
            The list of proteins and the cluster in which the proteins
            come from.
        """
        proteins: set[str] = set()
        for index, ordered_cluster in enumerate(order):
            # Don't get anymore proteins from order if we already have no
            # more than 80
            if len(proteins) > 80:
                proteins.clear()
                protein_cluster = order[: index - 1]
                for cluster in protein_cluster:
                    for protein in info[cluster]["kde"][-1]:
                        proteins.add(protein)
                break

            # Collect all the proteins up to 80
            protein_set = info[ordered_cluster]["kde"][-1]
            for protein in protein_set:
                proteins.add(protein)

        return list(proteins), protein_cluster  # type: ignore

    def _kegg_clusters(self, collection: str, partition: DataFrame) -> None:
        """Generate the individual cluster AUC scores per MCCV fold.

        Parameters
        ----------
        collection
            The nanoparticle
        partition
            The training dataframe
        """
        np.random.seed(self.constant_options.constants.RANDOM_STATE.value)
        os.makedirs(f"{self.kegg.result_path}/{collection}/Clusters", exist_ok=True)

        with open("KEGG/cluster_proteins.joblib", "rb") as file:
            cluster_proteins = load(file)

        for _, clusters in cluster_proteins[collection].items():
            for cluster_name, proteins in clusters.items():
                df = partition[proteins]
                model = next(iter(self.hyperparameters)).model
                gridsearch_score = self.run_grid_search(
                    df, self.combo_options, self.hyperparameters, model=model  # type: ignore
                )

                results = {}
                for (_, data) in gridsearch_score:
                    for kde, folds in data.items():
                        results[kde] = folds

                results[self.kde_method]["proteins"] = proteins
                with open(f"{self.kegg.result_path}/{collection}/Clusters/{cluster_name}.joblib", "wb") as file:
                    dump(results, file)

    def _run_inner_loop(
        self,
        collection: str,
        partition: DataFrame,
        class_: int,
        cluster_num: int,
        proteins: Proteins,
    ) -> None:
        np.random.seed(self.constant_options.constants.RANDOM_STATE.value)
        df = partition[proteins]

        model = next(iter(self.hyperparameters)).model
        gridsearch_score = self.run_grid_search(df, self.combo_options, self.hyperparameters, model=model)

        results = {}
        for (_, data) in gridsearch_score:
            for kde, folds in data.items():
                results[kde] = folds

        # Results/biological/DP/none/1/1_clusters.joblib >> Replacement
        # for ih/ie_b/none >> def loop
        with open(
            f"{self.kegg.result_path}/{collection}/{self.kde_method}/{class_}/{cluster_num}_clusters.joblib",
            "wb",
        ) as f:
            dump(results, f)
