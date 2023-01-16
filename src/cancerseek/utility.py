import os
import pickle
import sys
from dataclasses import dataclass
from typing import cast
from typing import Sequence

import numpy as np
import pandas as pd
import sklearn.neighbors._base
from src._types import Frames
from src._types import KDEType
from src.cancerseek.constants import KDEMethod

sys.modules["sklearn.neighbors.base"] = sklearn.neighbors._base  # missingpy is not up to date with newer versions
from missingpy import MissForest
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import OrdinalEncoder


@dataclass
class DataFeatures:
    """A data model holding feature information."""

    features: Sequence[str] = (
        "AFP (pg/ml)",
        "Angiopoietin-2 (pg/ml)",
        "AXL (pg/ml)",
        "CA-125 (U/ml)",
        "CA 15-3 (U/ml)",
        "CA19-9 (U/ml)",
        "CD44 (ng/ml)",
        "CEA (pg/ml)",
        "CYFRA 21-1 (pg/ml)",
        "DKK1 (ng/ml)",
        "Endoglin (pg/ml)",
        "FGF2 (pg/ml)",
        "Follistatin (pg/ml)",
        "Galectin-3 (ng/ml)",
        "G-CSF (pg/ml)",
        "GDF15 (ng/ml)",
        "HE4 (pg/ml)",
        "HGF (pg/ml)",
        "IL-6 (pg/ml)",
        "IL-8 (pg/ml)",
        "Kallikrein-6 (pg/ml)",
        "Leptin (pg/ml)",
        "Mesothelin (ng/ml)",
        "Midkine (pg/ml)",
        "Myeloperoxidase (ng/ml)",
        "NSE (ng/ml)",
        "OPG (ng/ml)",
        "OPN (pg/ml)",
        "PAR (pg/ml)",
        "Prolactin (pg/ml)",
        "sEGFR (pg/ml)",
        "sFas (pg/ml)",
        "SHBG (nM)",
        "sHER2/sEGFR2/sErbB2 (pg/ml)",
        "sPECAM-1 (pg/ml)",
        "TGFa (pg/ml)",
        "Thrombospondin-2 (pg/ml)",
        "TIMP-1 (pg/ml)",
        "TIMP-2 (pg/ml)",
    )
    categorical_features: Sequence[str] = ("Race", "Sex", "Ω score", "Age")
    labels: Sequence[str] = ("Patient ID #", "Sample ID #", "Tumor type", "AJCC Stage")


class Data:
    """A class with methods performed on the cancerseek data.

    Attributes
    ----------
    data_path
        The default path location for loading the cancerseek data.
    """

    data_path: str = "cancerseek_data"

    def __init__(
        self,
        features: Sequence[str] | None = None,
        categorical_features: Sequence[str] | None = None,
        labels: Sequence[str] | None = None,
    ) -> None:
        _params = {
            "features": features,
            "categorical_features": categorical_features,
            "labels": labels,
        }

        self.features = DataFeatures(**{k: v for k, v in _params.items() if v})

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Loads the capstone data.

        Returns
        -------
            The labels and input of the capstone data.
        """
        data = pd.read_csv(f"{self.data_path}/table6.csv")

        # Now let's do a union to add Omega
        omega_data = pd.read_csv(f"{self.data_path}/table5.csv")
        omega = omega_data[["Sample ID #", "Ω score"]]
        data = pd.merge(data, omega, on="Sample ID #", how="left")

        # Then another union to add Patient info
        personal_data = pd.read_csv(f"{self.data_path}/table4.csv")
        personal_data.rename(columns={"Plasma sample ID #": "Sample ID #"}, inplace=True)

        patient = personal_data[["Sample ID #", "Race", "Sex", "Age"]]
        data = pd.merge(data, patient, on="Sample ID #", how="left")

        data_labels = data[list(self.features.labels)]
        data_input = data[["Patient ID #"] + list(self.features.features) + list(self.features.categorical_features)]

        data_input = data_input.replace({",": ""}, regex=True)
        data_input = data_input.replace({r"\*": ""}, regex=True)

        for col in [feature for feature in self.features.features if feature not in ["Age"]]:
            data_input[col] = data_input[col].astype(float)

        return data_labels, data_input

    def create_splits(self, data_labels: pd.DataFrame, num_splits: int) -> pd.DataFrame:
        """Splits the patient data.

        For training, use `num_splits=9`, and `num_splits=1` for testing.
        Samples from the same patient are kept together.

        Parameters
        ----------
        data_labels
            The data labels.
        num_splits
            The number of splits to create.

        Returns
        -------
            Split data labels based on `num_splits`.
        """
        patients = data_labels["Patient ID #"].drop_duplicates().reset_index()
        splits = pd.DataFrame(
            np.random.default_rng().integers(0, num_splits, size=(patients.shape[0])),
            columns=["cv_split"],
        )
        patient_splits = pd.concat([patients, splits], axis=1, ignore_index=True)
        patient_splits.columns = ["-1", "Patient ID #", "cv_split"]
        patient_splits = patient_splits[["Patient ID #", "cv_split"]]
        data_labels = pd.merge(data_labels, patient_splits)

        return data_labels

    def data_preprocess(self, cache_path: str, refresh_cache: bool) -> tuple[np.ndarray, np.ndarray]:
        """Cleans and preprocesses the data.

        Data is loaded, categorical type columns are encoded, data
        is imputed with MissForest and the clean data is saved.

        Parameters
        ----------
        cache_path
            The path to save the data to.
        refresh_cache
            To refresh the data found in the `cache_path`.

        Returns
        -------
            A tuple of sample ID's with the cleaned data.
        """
        os.makedirs(cache_path, exist_ok=True)
        x_combined_ordinal_path = f"{cache_path}/X_combined_ordinal.npz"
        sample_id_path = f"{cache_path}/sample_IDs.npz"

        if not refresh_cache and os.path.isfile(x_combined_ordinal_path):
            print(f"\nCache file found in {x_combined_ordinal_path}. Loading now...")
            print("\nCache file found in {}. Loading now...".format(sample_id_path))
            x_combined_ordinal = np.load(x_combined_ordinal_path)["arr_0"]
            samples_ids = np.load(sample_id_path, allow_pickle=True)["arr_0"]

            return samples_ids, x_combined_ordinal

        print("No cache found for X_combined_ordinal.")
        # Loading Data
        print("Loading data...")
        data_labels, data_input = self.load_data()
        print("Raw data loaded.\n")

        categorical_features = [feat for feat in self.features.categorical_features if feat not in ["Ω score", "Age"]]
        x_floats = data_input[list(self.features.features) + ["Ω score", "Age"]].to_numpy()
        x_cat = data_input[categorical_features].to_numpy()

        # Ordinally encoding categorial X values
        print("Ordinally encoding categorical input features (ethnicity, race)...")
        x_cat_encoder = OrdinalEncoder(handle_unknown="error")
        x_cat_encoder.fit(x_cat)
        x_cat_ordinal = x_cat_encoder.transform(x_cat)
        print("Done.\n")

        # Group all numerical columns for imputation.
        x_combined_ordinal = np.concatenate([x_floats, x_cat_ordinal], axis=1)
        x_df_combined_ordinal = pd.DataFrame(data=x_combined_ordinal)

        # Imputing using MissForest
        print("Performing imputation on combined input data with ordinal encoding...")
        imputer = MissForest()
        x_df_combined_ordinal = imputer.fit_transform(x_df_combined_ordinal)
        x_combined_ordinal = x_df_combined_ordinal
        print("Done.\n")

        print(f"Saving imputed dataset files to `{x_combined_ordinal_path}`...")
        np.savez(x_combined_ordinal_path, x_combined_ordinal)

        samples_ids = np.array(data_labels["Sample ID #"]).reshape(-1, 1)
        print("Saving sample IDs files to `{}`...".format(sample_id_path))
        np.savez(sample_id_path, samples_ids)

        print("Done.\n")
        return samples_ids, x_combined_ordinal

    def feature_preprocess(self, cache_path: str, refresh_cache: bool) -> tuple[np.ndarray, ...]:
        """Cleans and preprocess feature data.

        Parameters
        ----------
        cache_path
            The path to save the data to.
        refresh_cache
            To refresh the data found in the `cache_path`.

        Returns
        -------
            Tumor type, stage, and each set of categories.
        """
        y_tumor_path = f"{cache_path}/Y_tumor.npz"
        y_stage_path = f"{cache_path}/Y_stage.npz"

        if not refresh_cache and os.path.isfile(y_tumor_path) and os.path.isfile(y_stage_path):
            print(f"\nLabel cache file found in {y_tumor_path} and {y_stage_path}. Loading now...")
            y_tumor = np.load(y_tumor_path)["arr_0"]
            y_stage = np.load(y_stage_path)["arr_0"]
            print("Done.\n")

            tumor_categories_path = f"{cache_path}/tumor_categories.pkl"
            stage_categories_path = f"{cache_path}/stage_categories.pkl"

            print(
                f"\nLoading category labels pkl files from `{tumor_categories_path}` and `{stage_categories_path}`..."
            )

            with open(tumor_categories_path, "rb") as tumor_handle, open(stage_categories_path, "rb") as stage_handle:
                tumor_categories = pickle.load(tumor_handle)
                stage_categories = pickle.load(stage_handle)

            return y_tumor, y_stage, tumor_categories, stage_categories

        # Extracting & Encoding Labels
        print(
            f"\nNo label cache found in {y_tumor_path} and {y_stage_path}."
            "\nGenerating ordinal encoded label matrices now."
        )

        print("Loading raw data...")
        data_labels, _ = self.load_data()

        print("\nGenerating ordinal encoding for labels...")
        # Converting labels values to numpy array, still with string data
        labels_tumor_types = data_labels["Tumor type"].to_numpy()
        labels_stages = data_labels["AJCC Stage"].to_numpy()

        # One-hot encoding labels
        tumor_ord_encoder = OrdinalEncoder()
        stage_ord_encoder = OrdinalEncoder()

        tumor_ord_encoder.fit(labels_tumor_types.reshape(-1, 1))
        stage_ord_encoder.fit(labels_stages.reshape(-1, 1))

        y_tumor = tumor_ord_encoder.transform(labels_tumor_types.reshape(-1, 1))
        y_stage = stage_ord_encoder.transform(labels_stages.reshape(-1, 1))

        # Ensure no values are nan
        y_stage = np.nan_to_num(y_stage, nan=-1) + 1
        assert not np.isnan(y_stage).any()

        print("Caching ordinally encoded Y_tumor and y_stage matrices...")
        np.savez(y_tumor_path, y_tumor)
        np.savez(y_stage_path, y_stage)

        print("Caching ordinal encoding arrays for tumor and stage labels...")
        tumor_categories = tumor_ord_encoder.categories_[0]
        stage_categories = stage_ord_encoder.categories_[0]

        tumor_categories_path = f"{cache_path}/tumor_categories.pkl"
        stage_categories_path = f"{cache_path}/stage_categories.pkl"

        with open(tumor_categories_path, "wb") as tumor_handle, open(stage_categories_path, "wb") as stage_handle:
            pickle.dump(tumor_categories, tumor_handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(stage_categories, stage_handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Done.\n")
        return y_tumor, y_stage, tumor_categories, stage_categories


class Augmentation:
    """A class for performing Kernel Density Estimation."""

    def augment_train_set(
        self,
        train_data: np.ndarray,
        train_label: np.ndarray,
        kde_params: KDEMethod,
        seed: int,
    ) -> Frames:
        """Applies Kernel Density to draw synthetic samples from.

        Parameters
        ----------
        train_data
            The training data.
        train_label
            The training labels.
        kde_params
            The parameter settings for kernel density.
        seed
            The random seed.

        Returns
        -------
            A set of preprocessed data by Kernel Density Augmentation.
        """
        kde_name = cast(KDEType, kde_params.name.lower())
        kde_param_values = kde_params.value
        if kde_param_values == 0:
            df = pd.DataFrame(train_data)
            df["classification"] = pd.Series(train_label)
            return Frames(name=kde_name, data=df)

        # Sample new points from the data according to the ratio
        # defined in kde_params. Combining X, Y data into a single matrix.
        num_features = train_data.shape[1]
        xy_train_combined_ordinal = np.concatenate(
            [
                train_data,
                train_label.reshape(-1, 1),
            ],
            axis=1,
        )

        new_samples_by_target = self._kde_ratios(
            xy_train_combined_ordinal.shape[0], kde_param_values[1], kde_param_values[0]
        )
        targets_idx = np.unique(xy_train_combined_ordinal[:, -1].astype(int))
        split_list = []

        print("\nSplit data to train independent grid search CV.")
        for i, target in enumerate(targets_idx):
            xy_train_by_target = xy_train_combined_ordinal[xy_train_combined_ordinal[:, -1].astype(int) == target]
            kde = self._grid_search(xy_train_by_target)
            augmented_target = self._kde_sampling(
                kde, xy_train_by_target, new_samples_by_target[i], num_features, seed
            )
            print(augmented_target[0].shape, augmented_target[1].shape)
            split_list.append(augmented_target)

        print("\nJoin individual targets.")
        (
            train_data_augmented,
            train_label_augmented,
        ) = [np.concatenate(join_targets, axis=0) for join_targets in zip(*split_list)]

        df = pd.DataFrame(train_data_augmented)
        df["classification"] = pd.Series(train_label_augmented)
        return Frames(name=kde_name, data=df)

    def _kde_ratios(self, input_size: int, distribution: tuple[int] = (1,), size: int = 5) -> list[int]:
        """Define the ratio for KDE.

        Parameters
        ----------
        input_size
            Original number of samples.

        distribution
            Proportion of samples to draw, by class, default = (1,)

        size
            Augmentation factor, default = 5

        Returns
        -------
            List containing the number of new samples to draw for each
            class.
        """
        total_samples = input_size * size
        ratios = []
        denom = sum(distribution)

        for el in distribution:
            ratios.append(int((total_samples / denom) * el))

        return ratios

    def _grid_search(self, df: pd.DataFrame) -> KernelDensity:
        """Bandwidth selection using a GridSearchCV.

        Parameters
        ----------
        df
            The dataframe used to fit the gridsearch.

        Returns
        -------
            The best estimator from the Kernel Density.
        """
        params = {"bandwidth": np.logspace(-1, 4, 50)}
        grid = GridSearchCV(KernelDensity(), params)
        grid.fit(df)

        return grid.best_estimator_  # type: ignore

    def _kde_sampling(
        self,
        kde: KernelDensity,
        xy_original: pd.DataFrame,
        num_new_samples: int,
        num_features: int,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gets and combines synthetic samples with original data.

        Parameters
        ----------
        kde
            The best kernel density estimator.
        xy_original
            The training data and labels.
        num_new_samples
            The number of new samples to draw.
        num_features
            The number of features in the training data.
        random_state, optional
            The random state to set for drawing samples, by default 42

        Returns
        -------
            A combined dataset with synthetic and original samples.
        """
        xy_new_data = kde.sample(num_new_samples, random_state=random_state)
        assert xy_new_data is not None
        print("Done drawing new samples in `xy_new_data`.")

        x_new_data = xy_new_data[:, :num_features]
        y_new_data = xy_new_data[:, -1]

        # Ensuring that y_new_data actually makes sense, obeys bounds of
        # original Ydata
        y_new_data = y_new_data.astype(int)
        y_new_data = np.clip(y_new_data, np.min(xy_original[:, -1]), np.max(xy_original[:, -1]))

        x_train_final = np.concatenate([xy_original[:, :num_features], x_new_data])
        y_train_final = np.concatenate([xy_original[:, -1], y_new_data])

        return x_train_final, y_train_final
