import glob
import os
import pickle
import re
import sys

import pandas as pd

sys.path.append(f"{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}\\Capstone")
base_path = "Results/validation"


def hyperparameter_best(cancer: str) -> None:
    """Finds and saves the best set of hyperparameters.

    This will run through each `cancer` with optimised
    parameters.

    Parameters
    ----------
    cancer
        The name of the cancer.
    """
    pkl_files = glob.glob(f"{base_path}/{cancer}/*/hyper_optimisation/*Validation.pkl")

    # Create directory where to store results
    res_dir = f"{base_path}/{cancer}/best_hyperparameters"
    os.makedirs(res_dir, exist_ok=True)

    data = []
    for pkl_file in pkl_files:
        p_list = re.split(r"/|\.", pkl_file.replace("\\", "/"))
        info = {"classifier": p_list[-2], "classification_mode": p_list[-4]}

        with open(pkl_file, "rb") as handle:
            gridsearch = pickle.load(handle)

        grid_info = [(valid_instance.get("auc"), valid_instance.get("params")) for valid_instance in gridsearch]
        grid_df = pd.DataFrame(grid_info, columns=["auc", "params"])

        best_params = grid_df.loc[grid_df["auc"].idxmax(), :]  # type: ignore

        info.update({"auc": best_params["auc"], "params": best_params["params"]})
        data.append(info)

    res_path = f"{res_dir}/best_hyperparameters.xlsx"
    writer = pd.ExcelWriter(res_path, engine="openpyxl")
    pd.DataFrame(data).to_excel(writer)
    writer.close()
