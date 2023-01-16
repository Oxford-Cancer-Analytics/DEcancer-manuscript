from os import getcwd
from os import listdir
from typing import Generator

import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import Constants

path = getcwd() + "/perseus_imputation"


def get_data() -> Generator:
    """Pre-imputed data is formatted into a shuffle-split dataframe.

    Yields
    -------
    GetDataType
        A tuple of data by name and the split dataframes.
    """
    csv_files = listdir(path)

    nanoparticles_dict = (
        (
            f.split("-")[0],
            pd.read_csv(
                "./perseus_imputation/" + f,
                index_col=["spion", "classification", "patients"],
            ),
        )
        for f in csv_files
    )

    # Split data into test and training sets in a stratified way.
    # First is the training set, second is the test set.

    for name, nano_df in nanoparticles_dict:
        yield name, train_test_split(
            nano_df,
            test_size=31,
            train_size=110,
            random_state=Constants.RANDOM_STATE.value,
            shuffle=True,
            stratify=nano_df.index.get_level_values(level="classification"),
        )


def control_data() -> Generator:
    """Pre-imputed data is formatted into a shuffle-split dataframe.

    The depleted plasma proteins are removed from the control data.

    Yields
    -------
    GetDataType
        A tuple of data by name and the split dataframes.
    """
    csv_files = listdir(path)
    dp = list(
        pd.read_csv(
            "./perseus_imputation/DP-imputed_data.csv",
            index_col=["spion", "classification", "patients"],
        ).columns
    )
    nanoparticles_dict = (
        (
            f.split("-")[0],
            pd.read_csv(
                "./perseus_imputation/" + f,
                index_col=["spion", "classification", "patients"],
            ),
        )
        for f in csv_files
        if "DP" not in f
    )

    for nano, df in nanoparticles_dict:
        diff = list(sorted(list(set(df.columns).difference(dp))))
        df = df[diff]
        yield nano, train_test_split(
            df,
            test_size=31,
            train_size=110,
            random_state=Constants.RANDOM_STATE.value,
            shuffle=True,
            stratify=df.index.get_level_values(level="classification"),
        )
