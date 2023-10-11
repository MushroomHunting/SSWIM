import numpy as np

import os
import pandas as pd

import seaborn as sbn
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import rfflib.utils.math_pytorch as mathtorch
import rfflib.utils.math_numpy as mathnp

matplotlib.rcParams.update(
    {'font.size': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42,
     'legend.fontsize': 8, 'image.cmap': "viridis"})
sbn.set(font_scale=0.9)
sbn.set_context(rc={'lines.markeredgewidth': 0.25})
sbn.set_style("whitegrid")

from tqdm import trange
from rfflib.utils.colour import named_CB as CBCOLS

from pathlib import Path

home = str(Path.home())  # logging path
log_path = f"{home}/experiment_results/perf_overfit"

all_datasets = {
    0: "elevators",
    1: "airfoil_noise",
    2: "concrete_compressive",
    3: "parkinsons_total",
    4: "bike_sharing_hourly",
    5: "ct_slice",
    6: "superconductivity",
}

dataset_name_short = {
    "elevators": "elevators",
    "airfoil_noise": "airfoil",
    "concrete_compressive": "concrete",
    "parkinsons_total": "parkinsons",
    "bike_sharing_hourly": "bike_hourly",
    "ct_slice": "ct_slice",
    "superconductivity": "super"
}

N_repeats = 10  # To verify each file actually has 10 rows
N_warplayers = 1

if __name__ == "__main__":
    for key in all_datasets.keys():
        dataset_name = all_datasets[key]
        # nlml_means = [] # each item corresponds to _all_ training iteration
        # nlml_stds = []
        # rmse_means = []
        # rmse_stds = []
        # mnlp_means = []
        # mnlp_stds = []
        # column_names = ["MSE_orig",
        #                 "RMSE_orig",
        #                 "MNLP_orig",
        #                 "MSE_norm",
        #                 "RMSE_norm",
        #                 "MNLP_norm"]
        result_filepath_nlml = os.path.join(log_path, f"{dataset_name}_sswim{1}_epochs{150}_nlml.csv")
        result_filepath_rmse = os.path.join(log_path, f"{dataset_name}_sswim{1}_epochs{150}_rmse.csv")
        result_filepath_mnlp = os.path.join(log_path, f"{dataset_name}_sswim{1}_epochs{150}_mnlp.csv")

        df_nlml = pd.read_csv(result_filepath_nlml)
        df_rmse = pd.read_csv(result_filepath_rmse)
        df_mnlp = pd.read_csv(result_filepath_mnlp)
        assert df_nlml.shape[0] == N_repeats
        assert df_rmse.shape[0] == N_repeats
        assert df_mnlp.shape[0] == N_repeats
        nlml_means = df_nlml.mean(axis=0)
        rmse_means = df_rmse.mean(axis=0)
        mnlp_means = df_mnlp.mean(axis=0)
        nlml_stds = df_nlml.std(axis=0)
        rmse_stds = df_rmse.std(axis=0)
        mnlp_stds = df_mnlp.std(axis=0)

        epochs = df_nlml.columns.astype(int)

        # rmse_means.append(means["RMSE_norm"])
        # rmse_stds.append(stds["RMSE_norm"])
        # mnlp_means.append(means["MNLP_norm"])
        # mnlp_stds.append(stds["MNLP_norm"])

        nlml_means = np.array(nlml_means)
        rmse_means = np.array(rmse_means)
        mnlp_means = np.array(mnlp_means)
        nlml_stds = np.array(nlml_stds)
        rmse_stds = np.array(rmse_stds)
        mnlp_stds = np.array(mnlp_stds)

        THE_FIG = plt.figure(dpi=200, figsize=(3, 6))
        ax = plt.subplot(3, 1, 1)
        ax.plot(epochs, nlml_means, c=CBCOLS["green"], lw=1.0)
        plt.fill_between(epochs, nlml_means - 1 * nlml_stds, nlml_means + 1 * nlml_stds, color=CBCOLS["green"], alpha=.2)
        ax.set_title(f"{dataset_name_short[dataset_name]}")
        # ax.set_xlabel("Epoch")
        # ax.set_xscale('log')
        ax.set_ylabel("NLML")

        ax = plt.subplot(3, 1, 2)
        ax.plot(epochs, rmse_means, c=CBCOLS["pink"], lw=1.0)
        plt.fill_between(epochs, rmse_means - 1 * rmse_stds, rmse_means + 1 * rmse_stds, color=CBCOLS["pink"], alpha=.2)
        # ax.set_title(f"{dataset_name_short[dataset_name]}")
        # ax.set_xlabel("Epoch")
        # ax.set_xscale('log')
        ax.set_ylabel("RMSE")

        ax = plt.subplot(3, 1, 3)
        ax.plot(epochs, mnlp_means, c=CBCOLS["blue_medium"], lw=1.0)
        plt.fill_between(epochs, mnlp_means - 1 * mnlp_stds, mnlp_means + 1 * mnlp_stds, color=CBCOLS["blue_medium"], alpha=.2)
        # ax.set_title(f"{dataset_name_short[dataset_name]}")
        ax.set_xlabel("Epoch")
        # ax.set_xscale('log')
        ax.set_ylabel("MNLP")

        THE_FIG.tight_layout()

        fig_path_pdf = os.path.join(log_path, "figs/", f"{dataset_name}_sswim{1}_epochs150.pdf")
        fig_path_png = os.path.join(log_path, "figs/", f"{dataset_name}_sswim{1}_epochs150.png")

        plt.savefig(fig_path_pdf, pad_inches=0.01)
        plt.savefig(fig_path_png, pad_inches=0.01)
