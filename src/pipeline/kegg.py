import os
from functools import cached_property
from typing import cast
from typing import Mapping
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import dump
from joblib import load
from pandas import DataFrame

from .._types import KDEType


class KEGG:
    """KEGG pathway analysis for the biological pipeline."""

    def __init__(self, collections: Sequence[str], kde_methods: Sequence[KDEType]) -> None:
        self.base_path = "KEGG"
        self.collections = collections
        self.kde_methods = kde_methods
        self.result_path = "Results/biological"

    @property
    def files(self) -> list[str]:
        """Retrieves all the files from the KEGG directory."""
        return [f for f in os.listdir(self.base_path) if "~$" not in f and "annotation" in f]

    @property
    def pathway_structure(self) -> DataFrame:
        """The KEGG structure, ordered by levels.

        Returns
        -------
            The pathway structure.
        """
        file_name = "KEGG Pathway_structure.xlsx"
        return pd.read_excel(f"{self.base_path}/{file_name}", names=["name", "level"], header=0)

    @cached_property
    def read_excel_files(self) -> list[tuple[DataFrame, str]]:
        """Excel file reader, cached.

        Returns
        -------
            The cached and preprocessed dataframe matched with the
            filename.
        """
        excel = []
        for filename in self.files:
            df = pd.read_excel(f"{self.base_path}/{filename}", usecols="ER:MS")
            df.columns = [
                list(df.columns)[0],  # patient Id
                *list(col.split("_")[1].lower() for col in df.columns[1:]),  # kegg proteins
            ]
            excel.append((df, filename))

        return excel

    def cluster_combinations(self) -> DataFrame:
        """The cluster combinations for each configuration.

        Returns
        -------
            The cluster combinations.
        """
        file_name = "Combination of Clusters_Biology based dimensionality reduction.xlsx"
        return pd.read_excel(f"{self.base_path}/{file_name}")

    def clusters(
        self, combinations: DataFrame | None = None, structure: DataFrame | None = None
    ) -> dict[str, list[str]]:
        """Get a dictionary of all cluster proteins.

        The cluster proteins are matched with each biological
        configuration.

        Parameters
        ----------
        combinations
            The cluster combinations for each configuration, by default
            None
        structure
            The KEGG structure, ordered by levels, by default None

        Returns
        -------
            A dictionary ordered by cluster with the KEGG labels as values
        """
        if combinations is None:
            combinations = self.cluster_combinations()  # pragma: no cover

        if structure is None:
            structure = self.pathway_structure  # pragma: no cover

        cluster_dict: dict[str, list] = {}

        for _, col in combinations.iteritems():
            # Remove NaN
            clusters = [c for c in col.values if isinstance(c, (str, np.str_))]
            index_1 = list(structure.index[structure["level"] == 1.0])
            index_2 = list(structure.index[structure["level"] == 2.0])

            for cluster in clusters:
                cluster = cluster.lower()
                for index, row in structure.iterrows():
                    level = row["level"]
                    name = row["name"]

                    # skipping blank rows
                    if (type(name) not in (str, np.str_) or np.isnan(level)) or (level == 3.0):
                        continue

                    name = " ".join(name.split(" ")[1:]).lower()
                    if cluster in cluster_dict:
                        continue

                    index = cast(int, index)
                    if name == cluster:
                        if level == 1.0:
                            names = self._cluster_names(index, index_1, structure)
                        else:
                            names = self._cluster_names(index, index_2, structure)
                        cluster_dict[cluster] = names
                    else:
                        continue

        with open(f"{self.base_path}/clusters.joblib", "wb") as path:
            dump(cluster_dict, path)
        return cluster_dict

    def split_cluster_proteins(
        self, cluster_combos: dict[str, list[str]] | None = None, **kwargs: Mapping[str, DataFrame]
    ) -> dict[str, dict]:
        """Splits the cluster proteins into their configurations.

        Parameters
        ----------
        cluster_combos
            The cluster combinations with KEGG levels, by default None
        **kwargs
            Options that are forwarded to `KEGG.clusters()`

        Returns
        -------
            A matched protein dictionary, mapping the KEGG proteins to each
            cluster.
        """
        if cluster_combos is None:
            cluster_combos = self.clusters(**kwargs)

        collection_classification: dict[str, dict] = {collection: {} for collection in self.collections}

        for df, filename in self.read_excel_files:
            collection = filename.split("-")[0]
            print(f"File = {collection}")
            collection_classification[collection] = {}

            for i, (_, col) in enumerate(self.cluster_combinations().iteritems()):
                collection_classification[collection][i + 1] = {}
                # Remove NaN values
                clusters = [c for c in col.values if isinstance(c, (str, np.str_))]

                for cluster_ in clusters:
                    cluster_ = cluster_.lower()
                    cluster_proteins = cast(list, cluster_combos.get(cluster_))
                    proteins: list[str] = []

                    items = list(set(cluster_proteins))
                    for kegg in items:
                        try:
                            data = df[kegg.lower()]
                        except KeyError:
                            # Not all proteins will be in the dataframe
                            continue

                        data = pd.concat([pd.Series(df["T: patients"]), data], axis=1)
                        data = data[data.iloc[:, 1] == 1]  # Get only the 1s
                        values = data["T: patients"].values
                        proteins.extend(values)
                    # Remove duplicate proteins within a cluster but keep
                    # dupes between clusters
                    proteins = list(set(proteins))
                    proteins.sort()
                    collection_classification[collection][i + 1][cluster_] = proteins

        with open(f"{self.base_path}/cluster_proteins.joblib", "wb") as path:
            dump(collection_classification, path)
        return collection_classification

    def average_clusters(
        self, classification: dict[str, dict] | None = None, **kwargs: Mapping[str, DataFrame]
    ) -> dict[str, dict]:
        """An average auc score will be calculated at each cluster.

        The averages are the initial cluster ranking.

        Parameters
        ----------
        classification
            The resulting dictionary of all classification protein
            clusters, by default None
        **kwargs
            Options that are forwarded to `KEGG.clusters()`

        Returns
        -------
            A more compact classification dictionary
        """
        if classification is None:
            classification = self.split_cluster_proteins(**kwargs)  # type: ignore[arg-type]

        averaged_classes: dict[str, dict] = {}
        for collection, classes in classification.items():
            averaged_classes[collection] = {}
            os.makedirs(f"{self.result_path}/{collection}/Clusters/", exist_ok=True)
            cluster_names = os.listdir(f"{self.result_path}/{collection}/Clusters/")
            for class_num, clusters in classes.items():
                averaged_classes[collection][class_num] = {}
                for cluster_name, proteins in clusters.items():
                    averaged_classes[collection][class_num][cluster_name] = {}
                    try:
                        cluster_file_name = next(
                            filter(
                                lambda x: " ".join(x.lower().split(".")[:-1]) == cluster_name,
                                cluster_names,
                            )
                        )
                        with open(
                            f"{self.result_path}/{collection}/Clusters/{cluster_file_name.lower()}",
                            "rb",
                        ) as file:
                            results = load(file)
                    except (FileNotFoundError, StopIteration):
                        # No proteins for that cluster
                        continue

                    for kde in self.kde_methods:
                        if kde not in results:
                            continue

                        aucs = []
                        for _, props in results[kde].items():
                            aucs.extend(props["auc"])

                        avgs = np.mean(aucs), np.std(aucs), len(proteins), proteins
                        averaged_classes[collection][class_num][cluster_name][kde] = avgs

        return averaged_classes

    def split_classes(self, classification: dict[str, dict] | None = None, **kwargs: Mapping[str, DataFrame]) -> None:
        """Splits the averaged classification into own class files.

        Parameters
        ----------
        classification
            The dictionary of all averaged classification protein
            clusters, by default None
        **kwargs
            Options that are forwarded to `KEGG.clusters()`
        """
        if classification is None:
            classification = self.average_clusters(**kwargs)  # type: ignore[arg-type]

        for collection, classes in classification.items():
            os.makedirs(f"{self.result_path}/{collection}/Classes", exist_ok=True)
            for class_num, clusters in classes.items():
                with open(f"{self.result_path}/{collection}/Classes/{class_num}.joblib", "wb") as f:
                    dump(clusters, f)

    def _cluster_names(self, index: int, indexes: list[int], structure: DataFrame) -> list[str]:
        """Helper function which gives the names of the cluster.

        Parameters
        ----------
        index
            The current index
        indexes
            The list of indexes that match a specific level
        structure
            The KEGG structure, ordered by levels

        Returns
        -------
            A sorted list of cluster names
        """
        names = []
        between = structure.iloc[index : self._get_next(index, indexes, structure), :]
        between = between[between.level == 3.0].dropna()
        names.extend(between.name.values)  # type: ignore

        return sorted(list(set(names)))

    def _get_next(self, value: int, values: list[int], structure: DataFrame) -> int:
        """Helper function to return the next index.

        Parameters
        ----------
        value
            The current index value
        values
            A list of index values matching up to the level
        structure
            The KEGG structure, ordered by levels

        Returns
        -------
            The next index level, or end of the dataframe
        """
        i = values.index(value)
        try:
            return values[i + 1 : i + 2][0]
        except IndexError:
            return structure.shape[0]
