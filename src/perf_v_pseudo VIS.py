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
sbn.set(font_scale=1.1)
sbn.set_context(rc={'lines.markeredgewidth': 0.25})
sbn.set_style("whitegrid")

from tqdm import trange
from rfflib.utils.colour import named_CB as CBCOLS

from pathlib import Path

home = str(Path.home())  # logging path
log_path = f"{home}/experiment_results/perf_v_pseudo"

all_datasets = [
    "ct_slice",
    "superconductivity",
    "elevators",
    "airfoil_noise",
    # "bike_sharing_hourly",
    "concrete_compressive",
    "parkinsons_total",

]

dataset_name_short = {
    "elevators": "elevators",
    "airfoil_noise": "airfoil",
    "concrete_compressive": "concrete",
    "parkinsons_total": "parkinsons",
    "bike_sharing_hourly": "bikeshare",
    "ct_slice": "ct slice",
    "superconductivity": "super"
}

N_repeats = 10  # To verify each file actually has 10 rows
N_pseudo_ = [20, 40, 80, 160, 320, 640, 1280]
N_warplayers = 1

THE_FIG = plt.figure(dpi=200, figsize=(18, 4))

if __name__ == "__main__":
    for i, dataset_name in enumerate(all_datasets):
        rmse_means = []
        rmse_stds = []
        mnlp_means = []
        mnlp_stds = []
        for N_pseudo in N_pseudo_:
            """ ----------- .csv Column names---------------------------------------------------
            - MSE_orig      -       Mean Squared Error in the original data units
            - RMSE_orig     -       Root Mean Squared Error in the original data units
            - MNLP_orig     -       Mean Negative Log Probability in original data units
            - MSE_norm      -       Mean Squared Error in the standardized data units
            - RMSE_norm     -       Root Mean Squared Error in the standardized data units
            - MNLP_norm     -       Mean Negative Log Probability in the original data units """
            result_filepath = os.path.join(log_path, f"{dataset_name}_sswim{N_warplayers}_epochs150_pseudo{N_pseudo}.csv")
            df = pd.read_csv(result_filepath)
            assert df.shape[0] == N_repeats
            means = df.mean(axis=0)
            stds = df.std(axis=0)

            rmse_means.append(means["RMSE_norm"])
            rmse_stds.append(stds["RMSE_norm"])
            mnlp_means.append(means["MNLP_norm"])
            mnlp_stds.append(stds["MNLP_norm"])

        rmse_means = np.array(rmse_means)
        rmse_stds = np.array(rmse_stds)
        mnlp_means = np.array(mnlp_means)
        mnlp_stds = np.array(mnlp_stds)

        ncols = len(all_datasets)

        ax = plt.subplot(2, ncols, i + 1)
        ax.plot(N_pseudo_, rmse_means, c=CBCOLS["red"], lw=1.0, label="mean")
        plt.fill_between(N_pseudo_, rmse_means - 1 * rmse_stds, rmse_means + 1 * rmse_stds, color=CBCOLS["red"], alpha=.2, label="$\pm 1$ std")
        ax.set_title(f"{dataset_name_short[dataset_name]}")#, fontsize=15)
        # ax.set_xlabel("$N_{pseudo}$")
        ax.set_xscale('log')
        if i == 0:
            ax.set_ylabel("RMSE")#, fontsize=15)
            plt.legend()


        ax = plt.subplot(2, ncols, ncols + i + 1)
        # ax.set_title(f"{dataset_name_short[dataset_name]}")
        ax.plot(N_pseudo_, mnlp_means, c=CBCOLS["blue"], lw=1.0, label="mean")
        plt.fill_between(N_pseudo_, mnlp_means - 1 * mnlp_stds, mnlp_means + 1 * mnlp_stds, color=CBCOLS["blue"], alpha=.2, label="$\pm 1$ std")
        ax.set_xlabel("$N_{pseudo}$", fontsize=18)
        ax.set_xscale('log')
        if i == 0:
            ax.set_ylabel("MNLP")#, fontsize=15)
            plt.legend()

    THE_FIG.tight_layout(pad=0.01)

    fig_path_pdf = os.path.join(log_path, "figs/", f"perf_v_pseudo.pdf")
    fig_path_png = os.path.join(log_path, "figs/", f"perf_v_pseudo.png")
    # fig_path_pdf = os.path.join(log_path, "figs/", f"perf_v_pseudo_sswim{N_warplayers}_epochs150.pdf")
    # fig_path_png = os.path.join(log_path, "figs/", f"perf_v_pseudo_sswim{N_warplayers}_epochs150.png")

    plt.savefig(fig_path_pdf, pad_inches=0.01)
    plt.savefig(fig_path_png, pad_inches=0.01)
