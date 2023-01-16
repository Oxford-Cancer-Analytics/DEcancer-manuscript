import glob
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from src.cancerseek.constants import FigureFlag
from src.cancerseek.figures import _FiguresValidation
from src.cancerseek.figures import filter_files
from src.cancerseek.figures import sort_files
from src.constants import PipelineStage
from src.data_models import FigureOptions
from src.pipeline import combo_pipelines

NANOPARTICLES = False
np.random.seed(0)

if NANOPARTICLES:
    from src.figures import Figures

    fig, ax = plt.subplots(figsize=(12, 12))
    spions = ["DP", "SP003", "SP006", "SP007", "SP333", "SP339"]
    for spion in spions:
        pipes = combo_pipelines([spion])
        stage = PipelineStage.TESTING

        for pipe in pipes:
            figure_options = FigureOptions(pipe, final=True, repeat=False)  # testing
            figures = Figures(ax, fig, figure_options)
            # figure_options = FigureOptions(pipe, repeat=False)  # control
            # figure_options = FigureOptions(pipe)  # validation

            if stage == PipelineStage.VALIDATION:
                figures.validation.create_figures(pipe.combo_options.collection, pipe.combo_options.data.training)
            elif stage == PipelineStage.TESTING:
                figures.testing.create_figures(pipe.combo_options.collection, pipe.combo_options.data)
            else:
                break

            figures.options.repeat = True
            plt.cla()
else:
    from src.cancerseek.figures import Figures

    figure_flag = FigureFlag.VALIDATION

    pkl_files = (
        filter_files(
            glob.glob("Results/test/*/*/*/*Test*.pkl"),
            "Breast",
            class_type="cancer-healthy",
            model_filter=["fullModel_all"],
        )
        if figure_flag == FigureFlag.TEST
        else sort_files(
            filter_files(
                glob.glob("Results/validation/*/*/0/*/*Validation*.pkl"),
                "Breast",
                class_type=["cancer-healthy"],
                model_filter=["fullModel_all", "proteinsOnly_postTTest"],
            )
        )
    )

    # Create only one figure and re-use instead of creating lots of figures
    fig, ax = plt.subplots(figsize=(8, 8))

    groups = []
    for pkl_file in pkl_files:
        # Creates an instance of a Figure class for each file
        figures = Figures(fig, ax)
        pkl_file = pkl_file.replace("\\", "/")
        print(pkl_file)

        if figure_flag == FigureFlag.VALIDATION:
            model_name = pkl_file.split("/")[-1].split(".")[0].split("Validation")[0]
            res = figures.validation.run_from_pickle(pkl_file, model_name)
            groups.append(res)
        elif figure_flag == FigureFlag.TEST:
            res = figures.testing.run_from_pickle(pkl_file)
            groups.append(res)

    if figure_flag == FigureFlag.VALIDATION:
        # Define a new figure for each group of 4
        fig, axes = plt.subplots(3, 2, figsize=(13, 13), sharex=True, sharey=True)
        axes = [ax for axesi in axes for ax in axesi]
        letters = ["a", "b", "c", "d", "e", "f"]

        class Shared:
            """Dummy Shared model."""

            def __init__(self):
                self.last_label = ""

        shared = Shared()

        # Create groups of 4 so all models get added onto same figure
        groups = [list(split) for split in np.array_split(groups, int(len(groups) / 4))]
        for group, ax, letter in zip(groups, axes, letters):
            best_model = max((m.auc_score, m.model_name) for m in group)[1]
            for model in group:
                model.shared = shared

            for index, model in enumerate(group):
                model = cast(_FiguresValidation, model)
                model.axes = ax
                model.grouped = True
                model.best_model = best_model

                # Will only apply the decor onto the figure once and
                # not overwrite itself
                if index % 4 == 3:
                    model.repeat = True

                model.create_roc_curve(model.recall_results, model.fp_results, letter)
    else:
        plt.cla()
        for i, model in enumerate(groups):
            model.grouped = True

            if i + 1 == len(groups):
                model.repeat = True

            model.create_roc_curve(model.y_pred, model.y_truth)
