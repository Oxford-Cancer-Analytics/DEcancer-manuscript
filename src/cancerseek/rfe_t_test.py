import glob
import os
import pickle
import re
from itertools import chain
from itertools import combinations
from itertools import product
from typing import Any
from typing import Iterable

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import t as t_distr


class TTest:
    """A class to run the T-test.

    This runs on the pickled recursive feature elimination data.
    """

    base_path = "Results/validation"

    def powerset(self, iterable: Iterable) -> chain[tuple[str, ...]]:
        """Produces all subsets of the `iterable`.

        Parameters
        ----------
        iterable
            An iterable of items.

        Returns
        -------
            The powerset of all items in the `iterable`.
        """
        iters = list(iterable)
        return chain.from_iterable(combinations(iters, i) for i in range(len(iters) + 1))

    def welch_t_test(
        self,
        best_auc: float,
        model_auc: float,
        se_x: float,
        se_y: float,
        num_x: int,
        num_y: int,
        two_sided: bool = True,
    ) -> np.ndarray:
        """Performs the Welch's t-test.

        Parameters
        ----------
        best_auc
            The best auc from recursive feature elimination.
        model_auc
            The auc from each protein in recursive feature elimination.
        se_x
            The standard error of x.
        se_y
            The standard error of y.
        num_x, num_y
            The number of samples for recursive feature elimination.
        two_sided, optional
            To use a two-sided t-test, by default True

        Returns
        -------
            The p-values.
        """
        std_err = np.sqrt(se_x**2 + se_y**2)
        t_statistic = (best_auc - model_auc) / std_err
        ddof = num_x + num_y - 2

        # Calculate the p-value
        if two_sided:
            p_value = (1.0 - t_distr.cdf(abs(t_statistic), ddof)) * 2
        else:
            p_value = 1.0 - t_distr.cdf(abs(t_statistic), ddof)

        return p_value

    def get_union_proteins(self, kde_bundle: list[dict[str, Any]]) -> dict[frozenset, set[str]]:
        """Gets the union set of features for each kernel density.

        Parameters
        ----------
        kde_bundle
            Information from the file path, dataframe and proteins.

        Returns
        -------
            A mapping over kernel density with the set of proteins.
        """
        all_combos = self.powerset([(res["KDE_method"], res["smallest_feature_set"]) for res in kde_bundle])
        ordered_combos = [tuple(zip(*el)) for el in all_combos if el]
        unions_dict = {frozenset(kde_method): set().union(*proteins) for kde_method, proteins in ordered_combos}

        return unions_dict

    def rfe_t_test(self, cancer: str) -> None:
        """Performs the full t-test on the feature elimination data.

        Parameters
        ----------
        cancer
            The type of cancer.
        """
        pkl_files = glob.glob(f"{self.base_path}/{cancer}/*/*/feature_elimination/RFE_*.pkl")

        # Create directory where to store results
        reuslt_directory = f"{self.base_path}/{cancer}/post_t_test"
        os.makedirs(reuslt_directory, exist_ok=True)

        data = self._get_data(pkl_files)

        classification_modes = {res["classification_mode"] for res in data}
        model_types = {res["model_type"] for res in data}

        for combo in product(classification_modes, model_types):
            kde_bundle = [res for res in data if (res["classification_mode"], res["model_type"]) == combo]

            result_path = os.path.join(reuslt_directory, "{}_{}_smallestSelected.xlsx".format(*combo)).replace(
                "\\", "/"
            )
            writer = pd.ExcelWriter(result_path, engine="xlsxwriter")
            print(result_path)
            for res in kde_bundle:
                if isinstance(res["selected_df"], DataFrame):
                    res["selected_df"].to_excel(writer, sheet_name=res["KDE_method"])

            # Union of features
            unions = self.get_union_proteins(kde_bundle)
            all_kde_proteins = frozenset(res["KDE_method"] for res in kde_bundle)
            union_features = unions[all_kde_proteins]
            union_features.discard("No proteins")
            pd.Series(list(union_features)).to_excel(writer, sheet_name="union_features")

            writer.close()

    def _get_data(self, pkl_files: list[str]) -> list[dict[str, Any]]:
        """Loops through the files to get the recursive feature data.

        Parameters
        ----------
        pkl_files
            A list of file paths.

        Returns
        -------
            A sequence of data from the file path, data from
            the dataframes and a set of proteins.
        """
        data = []

        # loop over the list of csv files
        for pkl_file in pkl_files:
            pkl_file = pkl_file.replace("\\", "/")

            # Retrieve information
            _, _, _, classification_mode, kde_method, *_, model_mode, _ = re.split(r"_|/|\.", pkl_file)
            info = {
                "classification_mode": classification_mode,
                "KDE_method": kde_method,
                "model_type": model_mode,
            }

            # read the pkl file
            with open(pkl_file, "rb") as handle:
                rfe_dict = pickle.load(handle)
            rfe_df = pd.DataFrame(rfe_dict).T

            info.update(zip(["selected_df", "smallest_feature_set"], self._t_test_from_df(rfe_df)))
            data.append(info)

        return data

    def _t_test_from_df(self, df: DataFrame) -> tuple[DataFrame, set[str]]:
        """Performs the t-test with the input data.

        Parameters
        ----------
        df
            The dataframe with the recursive feature elimination data.

        Returns
        -------
            The selected recursive feature elimination dataframe
            with p_values and a rejection column, together with
            the smallest set of proteins.
        """
        df = df.dropna(axis="index")
        df = df.astype({"protein_removed": "str", "auc": "float", "auc_sem": "float"})

        rfe_idx = df["auc"].idxmax()
        selected_df = df.loc[rfe_idx:, :]
        best_model: DataFrame = selected_df.loc[rfe_idx, ["auc", "auc_sem"]]  # type: ignore

        pvals = []
        for idx in selected_df.index:
            model = selected_df.loc[idx, ["auc", "auc_sem"]]
            p_value = self.welch_t_test(
                best_model.auc,
                model.auc,
                best_model.auc_sem,
                model.auc_sem,
                200,
                200,
                two_sided=False,
            )
            pvals.append(p_value)

        reject = [True if (p < 0.05) else False for p in pvals]

        selected_df = selected_df.assign(p_values=pvals, reject_null=reject)
        smallest_cardinality = selected_df[selected_df["reject_null"] == False].index.min()  # noqa: E712
        smallest_feature_set = set(selected_df.loc[smallest_cardinality:, "protein_removed"])

        return selected_df, smallest_feature_set
