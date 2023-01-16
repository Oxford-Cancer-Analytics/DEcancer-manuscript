from typing import Any
from typing import cast
from typing import Mapping
from typing import Sequence
from typing import TypeAlias

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from joblib import load
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from numpy import ndarray
from pandas import DataFrame
from pandas import Series
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from src.parameters import Parameter

from ._types import BaseEstimator
from ._types import KDEType
from ._types import Proteins
from .constants import AugmentationMethod
from .data_models import ComboOptions
from .data_models import FigureOptions
from .mccv import mc_cv
from .models import logisitic_regression
from .models import multi_layer_perceptron
from .models import random_forest
from .models import support_vector_machine


ModelEstimator: TypeAlias = (
    logisitic_regression.LogisticRegression
    | multi_layer_perceptron.MultiLayerPerceptron
    | random_forest.RandomForest
    | support_vector_machine.SupportVectorMachine
)


class Figures:
    """Produces ROC AUC figures."""

    def __init__(self, ax: Axes, figure: Figure, options: FigureOptions) -> None:
        self.axes = ax
        self.figure = figure
        self.options = options

        self.pca = _FiguresPCA(ax, figure)
        self.validation = _FiguresValidation(self, ax, options)
        self.testing = _FiguresTesting(self, ax, options)

    def create_roc_curve(
        self,
        classifiers: dict[str, Any],
        collection: str,  # noqa: U100
        num: int,
        prev_results: dict = {},  # noqa: U100
    ) -> None:
        """Creates the ROC curves.

        Parameters
        ----------
        classifiers
            The trained classifier models
        collection
            The collection member
        num
            The number of proteins
        prev_results, optional
            The results from previous function calls, by default {}
        """
        pipeline = self.options.pipe
        biological = pipeline.constant_options.biological
        control = pipeline.constant_options.control_data
        kde_method = pipeline.combo_options.kde_method.name.lower()

        repeat = self.options.repeat
        final = self.options.final

        np.random.seed(pipeline.constant_options.constants.RANDOM_STATE.value)

        plt.rcParams["font.family"] = "Times New Roman"  # type: ignore
        plt.rcParams.update({"font.size": 18})  # type: ignore
        plt.rc("axes", labelsize=20)

        mean_fpr = np.linspace(0, 1, 100)
        model_results: dict[str, dict] = {}

        for model, results in classifiers.items():
            model_results[model] = {"aucs": [], "tprs": []}
            for i, (estimator, x_valid, y_valid) in enumerate(results):
                print(model, i)
                model_name = estimator.__class__.__name__
                name = (
                    model_name
                    if model_name not in ["MLPClassifier", "SVC", "LogisticRegression"]
                    else "MultiLayerPerceptron"
                    if model_name == "MLPClassifier"
                    else "SupportVectorMachine"
                    if model_name == "SVC"
                    else "LogisticRegression (l2 penalty)"
                )

                viz = metrics.RocCurveDisplay.from_estimator(
                    estimator,
                    x_valid,
                    y_valid,
                    name=name,
                    pos_label="Early",
                    drop_intermediate=False,
                    alpha=0.8,
                    lw=2.5,
                    ax=self.axes,
                )

                self.axes.set(
                    xlim=[-0.05, 1.05],
                    ylim=[-0.05, 1.05],
                )
                self.axes.set_xlabel("1 - Specificity", fontsize=16)
                self.axes.set_ylabel("Sensitivity", fontsize=16)
                title = (
                    f"ROC curve showing Monte Carlo cross-validation performance of best {num}-protein classifier \n"
                    f"models in distinguishing lung cancer from cancer-free samples"
                )

                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                model_results[model]["aucs"].append(viz.roc_auc)
                model_results[model]["tprs"].append(interp_tpr)

        if repeat:
            classifier = {}
            for model, results in model_results.items():
                closest = min(results["aucs"], key=lambda x: abs(x - np.mean(results["aucs"])))

                for model_classifier, class_results in classifiers.items():
                    if model_classifier != model:
                        continue

                    index = results["aucs"].index(closest)
                    classifier[model_classifier] = [class_results[index]]

            self.options.repeat = False
            self.axes.cla()
            self.create_roc_curve(
                classifier,
                collection,
                num,
                prev_results=model_results,
            )

        if not repeat:
            results = model_results if final else prev_results
            self._roc_display(results, custom_labels=True, final=final, control=control)
            base_savepath = f"Results/{pipeline.combo_options.pipeline_type}/{collection}/figures_{collection}"
            if biological and final:
                file_string = f"{base_savepath}_biological_final.jpeg"
            elif biological:
                file_string = f"{base_savepath}_biological.jpeg"
            elif not biological and final:
                file_string = f"{base_savepath}_final.jpeg"
            elif control:
                file_string = f"{base_savepath}_control.jpeg"
            else:
                file_string = f"{base_savepath}_validation_{kde_method}.jpeg"
            self.figure.savefig(file_string, bbox_inches="tight", dpi=1000)

    def get_classifiers(
        self,
        model_class: ModelEstimator,
        df: DataFrame,
        combo_options: ComboOptions,
    ) -> list[tuple[BaseEstimator, DataFrame, Series]]:
        """Runs validation training.

        This produces the models and the training data associated
        with each of them.

        Parameters
        ----------
        model_class
            The class to use for model training.
        df
            The original data.
        combo_options
            A set of config options for each combination of parameters.

        Returns
        -------
            A list of models with validation data.
        """
        np.random.seed(0)
        classifiers = []
        options = {
            combo_options.preprocessor: True,
            combo_options.kde_method.name.lower(): combo_options.kde_method.value,
        }

        for index, (train_set, validation_set) in enumerate(
            mc_cv.split(df, y=df.index.get_level_values(level="classification"))
        ):
            print(index)
            kde_frames, x_valid, y_valid = model_class.preprocess(train_set, validation_set, **options)
            estimator: BaseEstimator

            clf = model_class.clf(
                random_state=self.options.pipe.constant_options.constants.RANDOM_STATE,
                **next(iter(model_class.hyperparameters)),
            )

            x_data = kde_frames.data.drop("classification", axis=1)
            y_data = kde_frames.data["classification"]

            estimator = clf.fit(x_data, y_data)
            classifiers.append((estimator, x_valid, y_valid))
        return classifiers

    def _roc_display(
        self,
        model_results: dict[str, dict],
        custom_labels: bool = True,
        final: bool = False,
        control: bool = False,
    ) -> None:
        """Produces figures for ROC_AUC graph.

        Parameters
        ----------
        model_results
            A list of all area under the curve values.
        custom_labels
            Whether to apply a custom formatted legend or use the default.
        final
            Uses the final test set and proteins.
        control
            Use the control data.
        """
        self.axes.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", alpha=0.8)

        handles, labels = self.axes.get_legend_handles_labels()
        if custom_labels:
            if control:
                labels = labels
            elif final:
                labels = [f"AUC: {labels[0][-7:-1]}"]
            else:
                labels = [
                    label[:-1] + f', STD = {np.std(results["aucs"]):.4f})'
                    for label, (_, results) in zip(labels, model_results.items())
                ]

        self.axes.legend(handles, labels, loc="lower right", fontsize=16)
        self.axes.grid(True)


class _FiguresPCA:
    """Figure methods for PCA plots."""

    def __init__(self, ax: Axes, figure: Figure) -> None:
        self.axes = ax
        self.figure = figure

    def create_figures(self, pca: PCA, df: DataFrame, x_comp: int = 1, y_comp: int = 2) -> None:
        """Produces figures for PCA.

        Parameters
        ----------
        pca
            The fitted PCA model.
        df
            The principal components.
        x_comp, y_comp
            The nth principal component.
        """
        if not ((0 < x_comp <= pca.n_components_) and (0 < y_comp <= pca.n_components_)):  # type: ignore
            raise ValueError("Invalid component!")

        self.axes.set_xlabel(f"Principal Component {x_comp}", fontsize=20)
        self.axes.set_ylabel(f"Principal Component {y_comp}", fontsize=20)
        self.axes.set_title("PCA", fontsize=20)

        targets = df["classification"]
        target_class = targets.unique()
        colors = ["r", "g"]
        markers = [".", "x"]
        for target, color, shape in zip(target_class, colors, markers):
            idx_to_keep = df["classification"] == target
            self.axes.scatter(
                df.loc[idx_to_keep, f"p_comp_{x_comp}"],  # type: ignore
                df.loc[idx_to_keep, f"p_comp_{y_comp}"],  # type: ignore
                c=color,
                marker=shape,  # type: ignore
                alpha=1,
                s=80,
            )

        self.axes.legend(target_class, fontsize=16)
        self.axes.grid()

        # Plot centroids and ellipses
        for target, color, shape in zip(target_class, colors, markers):
            idx_to_keep = df["classification"] == target
            centers = self._centroidnp(
                np.array(
                    [
                        df.loc[idx_to_keep, f"p_comp_{x_comp}"],
                        df.loc[idx_to_keep, f"p_comp_{y_comp}"],
                    ]
                ).T
            )
            self.axes.scatter(
                centers[0],
                centers[1],
                c=color,
                marker="D",  # type: ignore
                alpha=1,
                s=200,
            )
            self._confidence_ellipse(
                df.loc[idx_to_keep, f"p_comp_{x_comp}"],  # type: ignore
                df.loc[idx_to_keep, f"p_comp_{y_comp}"],  # type: ignore
                target,
                n_std=1.5,
                facecolor=color,
                edgecolor="b",
                alpha=0.1,
            )

        # Plot biplots projecting features as vectors onto PC axes
        # xvector = pca.components_[x_comp - 1]  # type: ignore
        # yvector = pca.components_[y_comp - 1]  # type: ignore
        # x_scaling = max(df[f"p_comp_{x_comp}"]) - min(df[f"p_comp_{x_comp}"])  # noqa: W505
        # y_scaling = max(df[f"p_comp_{y_comp}"]) - min(df[f"p_comp_{y_comp}"])  # noqa: W505

        # for i in range(len(xvector)):
        #     plt.arrow(
        #         0,
        #         0,
        #         xvector[i] * x_scaling,
        #         yvector[i] * y_scaling,
        #         color="b",
        #         width=0.0005,
        #         alpha=0.2,
        #         head_width=0.25,
        #     )

    def _centroidnp(self, arr: ndarray) -> tuple[float, float]:
        """Calculates the centroids for PCA.

        Parameters
        ----------
        arr
            An array of values.

        Returns
        -------
            PCA centroids.
        """
        length: int = arr.shape[0]
        sum_x: ndarray = np.sum(arr[:, 0])
        sum_y: ndarray = np.sum(arr[:, 1])

        return sum_x / length, sum_y / length  # type: ignore[return-value]

    def _confidence_ellipse(
        self, x_input: np.ndarray, y_input: np.ndarray, target: str, n_std: float = 3.0, **kwargs: object
    ) -> Ellipse:
        """Creates a covariance confidence ellipse.

        Parameters
        ----------
        x_input, y_input
            Input data.
        target
            The target.
        n_std
            The number of standard deviations to determine the ellipse's
            radius.
        **kwargs
            Forwarded to `~matplotlib.patches.Ellipse`

        Returns
        -------
            The plot axes.

        Raises
        ------
        ValueError
            If the size of `x_input` is not the same as `y_input`.
        """
        if x_input.size != y_input.size:
            raise ValueError("x and y must be the same size")

        cov = np.cov(x_input, y_input)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

        # Using a special case to obtain the eigenvalues of
        # this two-dimensional dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        wh_factor = 2 if target in ["cancer", "early", "late"] else 1.5
        ellipse = Ellipse((0, 0), width=ell_radius_x * wh_factor, height=ell_radius_y * wh_factor, **kwargs)

        # Calculating the standard deviation of x from the square root of
        # the variance and multiplying with the given number of standard
        # deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x_input)

        # calculating the standard deviation of y
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y_input)

        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

        ellipse.set_transform(transf + self.axes.transData)

        return self.axes.add_patch(ellipse)


class _FiguresValidation:
    def __init__(self, figures: Figures, ax: Axes, options: FigureOptions) -> None:
        self.axes = ax
        self.figures = figures
        self.options = options

    def create_figures(self, collection: str, partition: DataFrame) -> None:
        pipeline = self.options.pipe
        biological = pipeline.constant_options.biological
        control = pipeline.constant_options.control_data
        kde_method = cast(KDEType, pipeline.combo_options.kde_method.name.lower())
        combo_options = pipeline.combo_options

        file_string = f"Results/{pipeline.combo_options.pipeline_type}/Final_protein_sets_NB.joblib"
        with open(file_string, "rb") as f:
            selections: Proteins = load(f)[collection]
        selected_df: DataFrame = partition[selections]

        models = {}
        for model, (_, best) in pipeline.get_best_hyperparameters(kde_method, biological=biological, control=control)[
            collection
        ].items():
            best_param_values = best.iloc[0].name
            num = len(selections)
            if model != "random_forest":
                continue

            match model:
                case "support_vector_machine":
                    params = [
                        Parameter(SVC, name, [value]) for name, value in zip(best.index.names, best_param_values)
                    ]
                    model_class = support_vector_machine.SupportVectorMachine(
                        selected_df, combo_options, pipeline.constant_options, params=params
                    )
                case "random_forest":
                    params = [
                        Parameter(RandomForestClassifier, name, [value])
                        for name, value in zip(best.index.names, best_param_values)
                    ]
                    model_class = random_forest.RandomForest(
                        selected_df, combo_options, pipeline.constant_options, params=params
                    )  # type: ignore[assignment]
                case "logistic_regression":
                    params = [
                        Parameter(LogisticRegression, name, [value])
                        for name, value in zip(best.index.names, best_param_values)
                    ]
                    model_class = logisitic_regression.LogisticRegression(  # type: ignore[assignment]
                        selected_df, combo_options, pipeline.constant_options, params=params
                    )
                case "multi_layer_perceptron":
                    params = [
                        Parameter(MLPClassifier, name, [value])
                        for name, value in zip(best.index.names, best_param_values)
                    ]
                    model_class = multi_layer_perceptron.MultiLayerPerceptron(  # type: ignore[assignment]
                        selected_df, combo_options, pipeline.constant_options, params=params
                    )

            classifiers = self.figures.get_classifiers(
                model_class, selected_df, pipeline.combo_options  # type: ignore
            )
            models[model] = classifiers
        self.figures.create_roc_curve(models, collection, num)  # type: ignore


class _FiguresTesting:
    def __init__(self, figures: Figures, ax: Axes, options: FigureOptions) -> None:
        self.axes = ax
        self.figures = figures
        self.options = options

    def create_figures(self, collection: str, partitions: Sequence[DataFrame]) -> None:
        pipeline = self.options.pipe
        pipeline_type = pipeline.combo_options.pipeline_type

        biological = pipeline.constant_options.biological
        control = pipeline.constant_options.control_data
        kde_method = cast(KDEType, pipeline.combo_options.kde_method.name.lower())

        file_string = (
            f"Results/{pipeline_type}/Final_protein_sets_B.joblib"
            if biological
            else f"Results/{pipeline_type}/Final_protein_sets_NB.joblib"
        )

        with open(file_string, "rb") as f:
            selections: Proteins = load(f)[collection]
        train_df, test_df = partitions

        if not control:
            train_df, test_df = train_df[selections], test_df[selections]

        models = {}
        for model, (_, best) in pipeline.get_best_hyperparameters(kde_method, biological=biological, control=control)[
            collection
        ].items():
            if model != "random_forest":
                continue

            best_param = list(best.iloc[0].name)
            if isinstance(best_param[-2], type(np.nan)):
                best_param[-2] = None

            best_params = [
                Parameter(RandomForestClassifier, name, [value]) for name, value in zip(best.index.names, best_param)
            ]
            model_class = random_forest.RandomForest(
                train_df, pipeline.combo_options, pipeline.constant_options, best_params
            )
            num = train_df.shape[1]
            clf = model_class.clf().set_params(**next(iter(model_class.hyperparameters)))

            targets_train = train_df.index.get_level_values(level="classification").astype("category")
            targets_test = test_df.index.get_level_values(level="classification").astype("category")

            train = pd.concat(
                [
                    pd.Series(targets_train),
                    pd.DataFrame(train_df.values, columns=train_df.columns),
                ],
                axis=1,
            )
            testing = pd.concat(
                [
                    pd.Series(targets_test),
                    pd.DataFrame(test_df.values, columns=test_df.columns),
                ],
                axis=1,
            )
            x_test = testing.drop("classification", axis=1)
            y_test = testing["classification"].astype("category")

            options: Mapping[str, bool | tuple[int, int]] = {
                pipeline.combo_options.preprocessor: True,
                AugmentationMethod.NONE.name.lower(): AugmentationMethod.NONE.value,
            }
            frames = model_class.augment(train, options)

            x_data = frames.data.drop("classification", axis=1)
            y_data = frames.data["classification"]

            estimator = clf.fit(x_data, y_data)
            y_score = estimator.predict(x_test.values)

            # if not control:
            #     cf_matrix = metrics.confusion_matrix(y_test, y_score, labels=["Healthy", "Early"])
            #     cm_display = metrics.ConfusionMatrixDisplay(cf_matrix, display_labels=["Healthy", "Early"]).plot()
            #     cm_display.ax_.set_title(
            #         collection,
            #         fontdict={"fontsize": 15},
            #     )
            #     file_string = (
            #         f"Results/{pipeline_type}/{collection}/"
            #         f"{collection}_confusion_matrix{'_bio' if biological else ''}.jpg"
            #     )
            #     cm_display.figure_.savefig(file_string, bbox_inches="tight")

            models[model] = [(estimator, x_test, y_test)]
        self.figures.create_roc_curve(models, collection, num)  # type: ignore
