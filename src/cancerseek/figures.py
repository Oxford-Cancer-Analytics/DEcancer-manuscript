import os
import pickle
import re
from collections import namedtuple
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pandas import DataFrame
from pandas import Series
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from typing_extensions import Self

from .constants import ClassifierType

ep = 10 ** (-5)


class Figures:
    """A class to generate all figures."""

    cancer_annotation = "{} cancer."
    classifier_annotation = "{} classifier."
    class_annotation = {
        "cancer-healthy": "{} cancer samples vs healthy samples.",
        "cancer-otherCancers": "{} cancer samples vs other cancer samples.",
        "cancer-rest": "{} cancer samples vs any other samples.",
    }
    kde_annotation = {
        "0": "No KDE.",
        "5_4_1": "Imbalanced KDE in favour of base class.",
        "5_1_1": "KDE without class imbalance.",
        "5_1_4": "Imbalanced KDE in favour of lung cancer class.",
    }
    model_annotation = {
        "fullModel": "Epidemiological variables, Ω score and proteins included in the predictive algorithm.",
        "proteinsOnly": "Only proteins expression deployed in predictive algorithm.",
    }
    filter_annotation = {
        "bestRfe": "Proteins selected based on best AUC RFE step.",
        "all": "All 39 proteins selected in the algorithm.",
        "postTTest": "Best feature set according to T Test.",
    }

    def __init__(self, figure: Figure, axes: Axes, grouped: bool = False) -> None:
        self.figure: Figure = figure
        self.axes: Axes = axes
        self.grouped: bool = grouped

        self.testing = _FiguresTesting(self.axes, self, grouped=self.grouped)
        self.validation = _FiguresValidation(self.axes, self, grouped=self.grouped)
        self.utils = _FiguresUtility(self.axes, self)


class _FiguresUtility:
    """A utility class for figures.

    Methods which are used in both the testing
    and validation classes are grouped here.
    """

    def __init__(self, axes: Axes, figures: Figures) -> None:
        self.axes = axes
        self.figures = figures

    def add_j_score(
        self,
        mean_tpr: np.ndarray,
        mean_fpr: np.ndarray,
        mean_plot: list[Line2D] | None = None,
    ) -> int:
        """Calculates and adds the J-point to a figure plot.

        Parameters
        ----------
        mean_tpr
            The mean array of true-positive values.
        mean_fpr
            The mean array of false-positive values.
        mean_plot, optional
            A list of all points on a figure plot, by default None

        Returns
        -------
            The index of the J-point.
        """
        points = np.array([mean_fpr, mean_tpr]).T
        target = np.array([0, 1])
        point_index = (np.linalg.norm(target - points, axis=1)).argmin()  # type: ignore

        if mean_plot:
            # Now plot a single point with the same label
            self.axes.plot(
                mean_fpr[point_index],
                mean_tpr[point_index],
                "o",
                color=mean_plot[0].get_color(),
            )

        return point_index

    def add_specificity_point(
        self,
        points: list[Line2D],
        mean_fpr: np.ndarray,
        mean_tpr: np.ndarray,
        cancer: str,
        *,
        specificity_level: float = 0.99,
    ) -> float:
        """Calculates and adds the specificty point.

        Parameters
        ----------
        points
            A list of points from the figure plot.
        mean_fpr
            The mean array of false-positive values.
        mean_tpr
            The mean array of true-positive values.
        cancer
            The type of cancer.
        specificity_level, optional
            The level of specificity, by default 0.99

        Returns
        -------
            The sensitivity at the specified `specificity_level`.
        """
        spec_point = 1 - specificity_level

        if np.where(mean_fpr > spec_point)[0].size == 0:
            print(f"No {specificity_level}% Specificity point found!")
            return np.nan

        specificity_idx = np.where(mean_fpr > spec_point)[0][0]
        xp = mean_fpr[specificity_idx - 1 : specificity_idx + 1]
        yp = mean_tpr[specificity_idx - 1 : specificity_idx + 1]
        sensitivity_value = cast(float, np.interp(spec_point, xp, yp))

        # Now plot a single point with the same label if
        # cancer = pancancer
        if cancer == "pancancer":
            self.axes.plot(spec_point, sensitivity_value, "o", color=points[0].get_color())

        return sensitivity_value


class _FiguresTesting:
    """A class to generate the testing figures."""

    def __init__(self, axes: Axes, figures: Figures, grouped: bool = False) -> None:
        self.axes = axes
        self.figures = figures
        self.grouped = grouped
        self.repeat: bool = False

    def create_roc_curve(self, y_pred: np.ndarray, y_truth: np.ndarray, letter: str | None = None) -> None:
        """Generates the ROC figures.

        Parameters
        ----------
        y_pred
            The predicted values.
        y_truth
            The ground-truth values.
        letter
            The letter for a figure subplot.
        """
        auc_score = roc_auc_score(y_truth, y_pred)
        fpr, tpr, thresholds = roc_curve(
            y_truth[:, 1],
            y_pred[:, 1],
            pos_label=None,
            sample_weight=None,
            drop_intermediate=False,
        )

        points = self.axes.plot(  # noqa: F841
            fpr,
            tpr,
            label=f"ROC AUC: {auc_score:.4f}",
        )

        # J-score
        self.figures.utils.add_j_score(fpr, tpr)

        # optimal sensitivity / specificity values
        cut_off = thresholds[np.argmax(tpr - fpr)]
        sens = recall_score(y_truth[:, 1], y_pred[:, 1] > cut_off, pos_label=1)
        spec = recall_score(y_truth[:, 1], y_pred[:, 1] > cut_off, pos_label=0)

        # Add 0.99 specificity point
        self.figures.utils.add_specificity_point(points, fpr, tpr, self.cancer)  # type: ignore[has-type]

        j_point = sens + spec - 1
        print(j_point)

        if not self.repeat:
            # Decoration
            self.axes.grid(True, "major", "both")
            plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", alpha=0.8)
            self.axes.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])

        self.axes.legend(loc="lower right", fontsize=12)

        self.axes.set_xlabel("1 - Specificity", fontsize=12)
        self.axes.xaxis.tick_bottom()
        self.axes.set_ylabel("Sensitivity", fontsize=12)

        if letter:
            self.axes.text(-0.2, 1, letter)

        if self.grouped:
            self.results_path = "/".join(self.results_path.split("/")[:-4])  # type: ignore[has-type]
            plt.savefig(
                self.results_path + "/Figure_1.jpeg",  # type: ignore[has-type]
                bbox_inches="tight",
                dpi=1000,
            )
            return

        plt.savefig(
            self.results_path + f"ROC_{self.classification_mode}_{self.model_mode}_{self.filter_mode}.jpg",  # type: ignore[has-type]
            bbox_inches="tight",
        )

        plt.cla()

    def create_confusion_matrix(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        size: int | list[int],
        results_path: str,
        filename: str = "confusionMat",
    ) -> None:
        """Generates the confusion matrix figure.

        This is done iteratively for each model.

        Parameters
        ----------
        predictions
            The predictions from each model.
        labels
            The actual value labels.
        size
            The number of predictions.
        results_path
            The path to save the results.
        filename, optional
            The filename to append to `results_path`,
            by default "confusionMat"
        """
        if isinstance(size, int):
            size = [size, size]
        confusion_mat = np.zeros(size)
        tick_list = (
            ["Cancer free", "Lung cancer"] if size[0] == 2 else ["", "Cancer free", "Non-lung cancer", "Lung cancer"]
        )

        logits = predictions
        preds = np.argmax(logits, axis=1)
        true_res = np.argmax(labels, axis=1)
        i = 0
        while i < size[0]:
            j = 0  # noqa: VNE001
            while j < size[1]:
                matrix_index = ((preds == j) & (true_res == i)).astype(int).sum()
                confusion_mat[i, j] = matrix_index
                j += 1
            i += 1

        self.axes.matshow(confusion_mat, cmap="Blues", alpha=0.3)

        self.axes.set_xticks([0.0, 1.0], labels=tick_list)
        self.axes.set_yticks([0.0, 1.0], labels=tick_list)

        for i in range(size[0]):
            for j in range(size[1]):  # noqa: VNE001
                self.axes.text(
                    x=j,
                    y=i,
                    s=confusion_mat[i, j].astype(int),
                    va="center",
                    ha="center",
                    size="xx-large",
                )

        plt.xlabel("Predicted Class", fontsize=12)
        plt.ylabel("True Class", fontsize=12)
        plt.title("Confusion Matrix", fontsize=18)
        plt.savefig(
            results_path + f"{filename}_{self.cancer}_{self.classification_mode}_{self.model_mode}" + ".jpg",  # type: ignore[has-type]
            bbox_inches="tight",
        )
        plt.cla()

    def sensitivities_at_specificity(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        cancer_type: np.ndarray,
        stage: np.ndarray,
        specificity_level: float = 0.99,
    ) -> None:
        """Calculates sensitivity at a specific specificity.

        The results are saved in an excel file.

        Parameters
        ----------
        predictions
            The generated predictions from the model.
        labels
            The actual value labels.
        cancer_type
            The type of cancer.
        stage
            The stage of the cancer.
        specificity_level, optional
            The desired speicificty level, by default 0.99
        """
        cancer_dict = {
            0: "Breast",
            1: "Colorectum",
            2: "Esophagus",
            3: "Liver",
            4: "Lung",
            5: "Normal",
            6: "Ovary",
            7: "Pancreas",
            8: "Stomach",
        }

        # Transform arrays
        stage = stage.astype(int)
        cancer_type = cancer_type.astype(int)
        cancer_type = np.array([cancer_dict[key] for key in cancer_type])

        fpr, tpr, thresholds = roc_curve(
            labels[:, 1],
            predictions[:, 1],
            pos_label=None,
            sample_weight=None,
            drop_intermediate=False,
        )

        spec_point = 1 - specificity_level
        # Cancer sensitivities at approx. 0.99 specificity
        if np.where(fpr > spec_point)[0].size == 0:
            print(f"No {specificity_level}% Specificity point found!")
            return

        specificity_idx = np.where(fpr > spec_point)[0][0]
        overall_sensitivity_value = tpr[specificity_idx]
        thr = thresholds[specificity_idx]

        # Filter all positives (true positives and false negatives)
        pos_mask = labels[:, 1] >= thr
        cancer_pos = cancer_type[pos_mask]
        stage_pos = stage[pos_mask]

        # Filter true positives
        tp_mask = np.logical_and(predictions[:, 1] >= thr, pos_mask)
        cancer_tp = cancer_type[tp_mask]
        stage_tp = stage[tp_mask]

        # Combine stage and cancer type
        pos_data = pd.DataFrame(np.column_stack([stage_pos, cancer_pos]), columns=["Stage", "Cancer"])
        tp_data = pd.DataFrame(np.column_stack([stage_tp, cancer_tp]), columns=["Stage", "Cancer"])

        def generate_table(pos_data: DataFrame, tp_data: DataFrame, combination: list[str]) -> DataFrame | Series:
            if len(combination) == 2:
                count_pos = pos_data.value_counts(combination).unstack(fill_value=0)
                count_tp = tp_data.value_counts(combination).unstack(fill_value=0)
                sensitivities = count_tp.divide(count_pos)
            else:
                count_pos = pos_data.value_counts(combination).fillna(0).rename("Sensitivity")
                count_tp = tp_data.value_counts(combination).fillna(0).rename("Sensitivity")
                sensitivities = count_tp.divide(count_pos)

            return sensitivities

        writer = pd.ExcelWriter(
            self.results_path + f"Sensitivities_{self.classification_mode}_{self.model_mode}_{self.filter_mode}.xlsx",  # type: ignore[has-type]
            engine="openpyxl",
        )

        # Overall sensitivity
        pd.DataFrame([overall_sensitivity_value]).to_excel(writer, sheet_name="Overall sensitivity")

        # Sensitivities by Stage
        sen_stage = generate_table(pos_data, tp_data, ["Stage"])
        sen_stage.to_excel(writer, sheet_name="Stage")

        if self.cancer == "pancancer":  # type: ignore[has-type]
            # Sensitivities by Cancer
            sen_cancer = generate_table(pos_data, tp_data, ["Cancer"])
            sen_cancer.to_excel(writer, sheet_name="Cancer")

            # Sensitivities by Stage and Cancer
            sen_stage_cancer = generate_table(pos_data, tp_data, ["Stage", "Cancer"])
            sen_stage_cancer.to_excel(writer, sheet_name="Stage, Cancer")

        writer.close()

    def run_from_pickle(self, file_path: str) -> Self:  # type: ignore[valid-type]
        """Loads data from pickled files to generate testing figures.

        Parameters
        ----------
        file_path
            The file path of the pickled file.
        """
        with open(file_path, "rb") as handle:
            test_file = pickle.load(handle)

        # First generate output directory
        (
            *_,
            cancer,
            classification_mode,
            model_mode,
            filter_mode,
            classifier,
            _,
        ) = re.split(r"[\./_]", file_path)
        self.results_path = f"Results/figures/test/{cancer}/{classification_mode}/{model_mode}_{filter_mode}/"

        self.cancer = str(cancer)
        self.classifier = str(classifier)
        self.classification_mode = str(classification_mode)
        self.model_mode = str(model_mode)
        self.filter_mode = str(filter_mode)

        # Before producing results, create directory
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)

        # Now generate results
        for res in test_file:
            y_pred: np.ndarray = res["results"]["test_results"]["y_pred"]
            y_truth: np.ndarray = res["results"]["test_results"]["y_truth"]
            y_cancer_type: np.ndarray = res["results"]["source_test_labels"]
            y_stage: np.ndarray = res["results"]["stage_test_labels"]

            # Create confusion matrix
            self.create_confusion_matrix(y_pred, y_truth, y_pred.shape[1], self.results_path)  # noqa: W505

            # Create file of sensitivities at given specificity threshold
            self.sensitivities_at_specificity(y_pred, y_truth, y_cancer_type, y_stage)

            # Create ROC curve
            self.y_pred = y_pred
            self.y_truth = y_truth
            self.create_roc_curve(y_pred, y_truth)

        return self


class _FiguresValidation:
    """A class to generate the validation figures."""

    def __init__(self, axes: Axes, figures: Figures, grouped: bool = False) -> None:
        self.axes = axes
        self.figures = figures
        self.grouped = grouped
        self.repeat: bool = False
        self.best_model = ""

    def calculate_optimal_points(
        self, y_truth_sets: dict[int, np.ndarray], y_pred_sets: dict[int, np.ndarray]
    ) -> tuple[float, float, float]:
        j_points = []
        sensitivities = []
        specificities = []

        for (_, y_truth), (_, y_pred) in zip(y_truth_sets.items(), y_pred_sets.items()):
            fpr, tpr, thresholds = roc_curve(
                y_truth[:, 1],
                y_pred[:, 1],
                pos_label=None,
                sample_weight=None,
                drop_intermediate=False,
            )

            cut_off = thresholds[np.argmax(tpr - fpr)]
            specificity = recall_score(y_truth[:, 1], y_pred[:, 1] > cut_off, pos_label=0)
            sensitivity = recall_score(y_truth[:, 1], y_pred[:, 1] > cut_off, pos_label=1)

            j_point = sensitivity + specificity - 1
            j_points.append(j_point)
            sensitivities.append(sensitivity)
            specificities.append(specificity)

        return np.mean(j_points), np.mean(sensitivities), np.mean(specificities)

    def create_roc_curve(
        self,
        scaled_recall_results: dict[int, list[float]],
        scaled_fp_results: dict[int, list[float]],
        letter: str | None = None,
    ) -> None:
        """Generates the ROC figures.

        Parameters
        ----------
        scaled_recall_results
            A mapping of the recall results.
        scaled_fp_results
            A mapping of the false-positive results.
        letter
            The letter to use if a multi-graph image.
        """
        j_point, sensitivity, specificity = self.calculate_optimal_points(self.y_truth_sets, self.y_pred_sets)
        print(f"{self.model_name=}, {self.repeat=}, {j_point=}, {sensitivity=}, {specificity=}")

        # Plot mean ROC too
        fp_concat = np.array(list(scaled_fp_results.values())).T / 100
        fp_concat = np.insert(fp_concat, fp_concat.shape[0], 0, axis=0)
        fp_concat = np.insert(fp_concat, 0, 1, axis=0)
        mean_fpr = np.flip(np.mean(fp_concat, axis=1))

        recall_concat = np.array(list(scaled_recall_results.values())).T / 100
        recall_concat = np.insert(recall_concat, recall_concat.shape[0], 0, axis=0)
        recall_concat = np.insert(recall_concat, 0, 1, axis=0)
        mean_tpr = np.flip(np.mean(recall_concat, axis=1))

        self.axes.plot(  # noqa: F841
            mean_fpr,
            mean_tpr,
            label=f"{self.model_name} (AUC={self.auc_score:.4f}, STD={(self.auc_sem * np.sqrt(200)):.4f})",
        )

        if self.repeat:
            # Decoration
            self.axes.grid(True, "major", "both")
            self.axes.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", alpha=0.8)
            self.axes.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])

            self.axes.set_ylabel("Sensitivity", fontsize=13)
            self.axes.set_xlabel("1 - Specificity", fontsize=13)
            self.axes.xaxis.tick_bottom()

        if letter:
            self.axes.text(-0.2, 1, letter)

        # Add optimal points
        if self.best_model == self.model_name:
            last_label = f"Optimal sensitivity and specificity corresponding\nto the top left corner of ROC curve is {sensitivity * 100:.2f}% and\n{specificity * 100:.2f}%, respectively."
            self.shared.last_label = last_label  # type: ignore

        if self.repeat:
            self.axes.plot([], [], " ", label=self.shared.last_label)  # type: ignore

        self.axes.legend(loc="lower right", fontsize=9)

        if self.grouped:
            self.results_path = "/".join(self.results_path.split("/")[:-4])  # type: ignore[has-type]
            plt.savefig(
                self.results_path + "/Supplement Figure 1.jpeg",  # type: ignore[has-type]
                bbox_inches="tight",
                dpi=1000,
            )
            return

        if not self.grouped:
            plt.cla()

    def calculate_metrics(
        self,
        y_pred_sets: dict[int, np.ndarray],
        y_truth_sets: dict[int, np.ndarray],
        scaling: np.ndarray,
    ) -> dict[int, list[float]]:
        """Calculates the validation metrics.

        Metrics include the accuracy, recall, precision and
        false-positive rate.

        Parameters
        ----------
        y_pred_sets
            The y prediction data.
        y_truth_sets
            The y ground-truth data.
        scaling
            Scaling.

        Returns
        -------
            Grouped metrics
        """
        model_results = {}
        for key in y_pred_sets:
            logits = y_pred_sets[key]
            labels = np.argmax(y_truth_sets[key], axis=1)

            preds = np.argmax(np.multiply(logits, scaling), axis=1)
            accuracy = round(np.sum(preds == labels) / labels.shape[0] * 100, 2)
            recall = round(
                np.sum((preds == labels) & (preds == 1)) / (labels[labels == 1].shape[0] + ep) * 100,
                2,
            )
            precision = round(
                np.sum((preds == labels) & (preds == 1)) / (preds[preds == 1].shape[0] + ep) * 100,
                2,
            )
            fp_rate = round(
                np.sum((labels == 0) & (preds == 1)) / (labels[labels == 0].shape[0] + ep) * 100,
                2,
            )

            model_results[key] = [accuracy, recall, precision, fp_rate]

        return model_results

    def run_from_pickle(self, file_path: str, model_name: str) -> Self:  # type: ignore[valid-type]
        """Loads data from pickled files to generate validation figures.

        Parameters
        ----------
        file_path
            The file path of the pickled file.
        model_name
            The name of the classifier model.
        figures
            An instance of the parent class. This gives access to
            other figure functionality.

        Returns
        -------
            An instance of itself.
        """
        with open(file_path, "rb") as handle:
            pkl_file = pickle.load(handle)

        self._create_directories(file_path, model_name)

        for res in pkl_file:
            self.auc_score: float = res["roc_auc"]
            self.auc_sem: float = res["auc_sem"]

            # Now we're looking at the average of the ROC curves produced
            self.y_pred_sets = {}
            self.y_truth_sets = {}

            for index, res_set in enumerate(res["results"]):
                self.y_pred_sets[index] = res_set["valid_results"]["y_pred"]
                self.y_truth_sets[index] = res_set["valid_results"]["y_truth"]

            scaling = np.exp(np.linspace(-4, 4, 502))

            # make sure you always have identity scaling in results
            identity_index = np.argmin(np.abs(1 - scaling), axis=0)
            scaling[identity_index] = 1
            scaling = np.expand_dims(scaling, 1)

            # Configured for 2 CLASS!
            scaling = np.concatenate(
                [scaling, np.ones((scaling.shape[0], ClassifierType.CLASS_2.value))],
                axis=1,
            )
            scaled_precision_results: dict[int, list[float]] = {}
            scaled_recall_results: dict[int, list[float]] = {}
            scaled_fp_results: dict[int, list[float]] = {}

            i = 1
            while i < (scaling.shape[0] - 1):
                model_results = self.calculate_metrics(self.y_pred_sets, self.y_truth_sets, scaling[i, :])
                for key in model_results:
                    if key not in scaled_precision_results:
                        scaled_recall_results[key] = []
                        scaled_precision_results[key] = []
                        scaled_fp_results[key] = []

                    scaled_recall_results[key].append(model_results[key][1])
                    scaled_precision_results[key].append(model_results[key][2])
                    scaled_fp_results[key].append(model_results[key][3])
                i += 1

            self.recall_results = scaled_recall_results
            self.fp_results = scaled_fp_results

            if self.grouped:
                # create ROC curve
                self.create_roc_curve(
                    scaled_recall_results,
                    scaled_fp_results,
                )

        return self

    def _create_directories(self, file_path: str, model_name: str) -> str:
        """Sets some instance variables for use in other methods.

        Parameters
        ----------
        file_path
            The file path of the pickled file.
        model_name
            The name of the classifier model.

        Returns
        -------
            The path for results to be stored in.
        """
        self._set_meta_variables(file_path, model_name)

        # Before producing results, create directory
        results_path = os.path.join("Results", "figures", *re.split(r"/|\.", file_path)[1:-2], model_name)
        os.makedirs(os.path.dirname(f"{results_path}/"), exist_ok=True)

        self.results_path = results_path.replace("\\", "/")
        return self.results_path

    def _set_meta_variables(self, file_path: str, model_name: str) -> None:
        """Sets some instance variables for use in other methods.

        Parameters
        ----------
        file_path
            The file path of the pickled file.
        model_name
            The name of the classifier model.
        """
        (
            *_,
            cancer,
            classification_mode,
            kde_method,
            model_mode,
            filter_mode,
            classifier,
            _,
        ) = re.split(r"/|\.|_", file_path)

        self.cancer = cancer
        self.classification_mode = classification_mode
        self.kde_method = kde_method
        self.model_mode = model_mode
        self.filter_mode = filter_mode
        self.classifier = classifier
        self.model_name = model_name


def sort_files(files: list[str]) -> list[str]:
    """Sorts validation files into a specific order.

    This makes the figures compatible with nanoparticles figures.

    Parameters
    ----------
    files
        List of file paths.

    Returns
    -------
        A custom sorted list of file paths.
    """
    # Group them into 4
    groups = [list(split) for split in np.array_split(files, int(len(files) / 4))]

    order = {
        "MLP": 1,
        "LogReg": 2,
        "SVM": 3,
        "RandomForest": 4,
    }

    all_paths = []
    for group in groups:
        sorted_files = list(
            sorted(
                group,
                key=lambda x: order[
                    re.split(
                        r"(.*)(Validation|Test).pkl",
                        x.replace("\\", "/").split("/")[-1],
                    )[1]
                ],
            )
        )
        all_paths.extend(sorted_files)

    return all_paths


def filter_files(
    files: list[str],
    cancer_type: str | list[str],
    class_type: str | list[str] | None = None,
    model_filter: str | list[str] | None = None,
) -> list[str]:
    """Filters files based on specific parameters.

    This makes the figures compatible with nanoparticles figures.

    Parameters
    ----------
    files
        List of file paths.
    cancer_type
        The type of cancer.
    class_type
        The type of class used.
    model_filter
        The combination of model and filter used.

    Returns
    -------
        A custom filtered list of file paths.
    """
    file_opts = namedtuple("file_opts", ["cancer_type", "class_type", "model_filter", "model_name"])

    def split_file(file_path: str) -> file_opts:
        if "Test.pkl" in file_path:
            _, cancer, classification_mode, model_filter, model_name = re.split(r"\\", file_path)
        else:
            _, cancer, classification_mode, _, model_filter, model_name = re.split(r"\\", file_path)
        return file_opts(cancer, classification_mode, model_filter, model_name)

    def match_opts(file_path: str, option: str | list[str] | None) -> str | None:
        options = split_file(file_path)
        if isinstance(option, str):
            option = [option]

        if option is None:
            return file_path

        if options.cancer_type in [
            "Breast",
            "Colorectum",
            "Esophagus",
            "Liver",
            "Lung",
            "Ovary",
            "pancancer",
            "Stomach",
            "Pancreas",
        ] and any(opt in options.cancer_type for opt in option):
            return file_path
        elif options.class_type in [
            "cancer-healthy",
            "cancer-otherCancers",
            "cancer-rest",
        ] and any(opt in options.class_type for opt in option):
            return file_path
        elif options.model_filter in [
            "fullModel_all",
            "proteinsOnly_all",
            "proteinsOnly_postTTest",
        ] and any(opt in options.model_filter for opt in option):
            return file_path
        else:
            return None

    filtered_paths = filter(
        (lambda x: (match_opts(x, cancer_type) and match_opts(x, class_type) and match_opts(x, model_filter))),
        files,
    )

    return list(filtered_paths)
