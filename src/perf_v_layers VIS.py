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
log_path = f"{home}/experiment_results/perf_v_layers"

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
    # "bike_sharing_hourly": "bikeshare",
    "ct_slice": "ct slice",
    "superconductivity": "super"
}

N_repeats = 10  # To verify each file actually has 10 rows
N_warplayers_ = [0, 1, 2, 3]

if __name__ == "__main__":
    THE_FIG = plt.figure(dpi=200, figsize=(18, 4))
    for i, dataset_name in enumerate(all_datasets):
        rmse_means = []
        rmse_stds = []
        mnlp_means = []
        mnlp_stds = []
        for N_warplayers in N_warplayers_:
            # column_names = ["MSE_orig",
            #                 "RMSE_orig",
            #                 "MNLP_orig",
            #                 "MSE_norm",
            #                 "RMSE_norm",
            #                 "MNLP_norm"]
            result_filepath = os.path.join(log_path, f"{dataset_name}_sswim{N_warplayers}_{150}.csv")
            # result_filepath = os.path.join(log_path, f"{dataset_name}_sswim{N_warplayers}_{150}.csv")
            df = pd.read_csv(result_filepath)
            try:
                assert df.shape[0] == N_repeats
            except Exception as the_exception:
                print(dataset_name)
                print(N_warplayers)
                exit(0)
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

        # ax = plt.subplot(1, 2, 1)
        # ax.plot(N_warplayers_, rmse_means, c=CBCOLS["red"], lw=1.0)
        # plt.fill_between(N_warplayers_, rmse_means - 1 * rmse_stds, rmse_means + 1 * rmse_stds, color=CBCOLS["red"], alpha=.2)
        # ax.set_title(f"{dataset_name_short[dataset_name]}")
        # ax.set_xlabel("$N_{warping levels}$")
        # # ax.set_xscale('log')
        # ax.set_ylabel("RMSE")
        #
        # ax = plt.subplot(1, 2, 2)
        # ax.set_title(f"{dataset_name_short[dataset_name]}")
        # ax.plot(N_warplayers_, mnlp_means, c=CBCOLS["blue"], lw=1.0)
        # plt.fill_between(N_warplayers_, mnlp_means - 1 * mnlp_stds, mnlp_means + 1 * mnlp_stds, color=CBCOLS["blue"], alpha=.2)
        # ax.set_xlabel("$N_{warping levels}$")
        # # ax.set_xscale('log')
        # ax.set_ylabel("MNLP")

        ncols = len(all_datasets)

        ax = plt.subplot(2, ncols, i + 1)
        ax.plot(N_warplayers_, rmse_means, c=CBCOLS["red"], lw=1.0, label="mean")
        plt.fill_between(N_warplayers_, rmse_means - 1 * rmse_stds, rmse_means + 1 * rmse_stds, color=CBCOLS["red"], alpha=.2, label="$\pm 1$ std")
        ax.set_title(f"{dataset_name_short[dataset_name]}")#, fontsize=18)
        # ax.set_xlabel("$N_{pseudo}$")
        # ax.set_xscale('log')
        if i == 0:
            ax.set_ylabel("RMSE")#, fontsize=18)
            plt.legend()

        ax = plt.subplot(2, ncols, ncols + i + 1)
        # ax.set_title(f"{dataset_name_short[dataset_name]}")
        ax.plot(N_warplayers_, mnlp_means, c=CBCOLS["blue"], lw=1.0, label="mean")
        plt.fill_between(N_warplayers_, mnlp_means - 1 * mnlp_stds, mnlp_means + 1 * mnlp_stds, color=CBCOLS["blue"], alpha=.2, label="$\pm 1$ std")
        ax.set_xlabel("$N_{warping\ levels}$", fontsize=18)
        # ax.set_xscale('log')
        if i == 0:
            ax.set_ylabel("MNLP")#, fontsize=18)
            plt.legend()

    THE_FIG.tight_layout(pad=0.01)

    # fig_path_pdf = os.path.join(log_path, "figs/", f"{dataset_name}_0_to_3_layers.pdf")
    # fig_path_png = os.path.join(log_path, "figs/", f"{dataset_name}_0_to_3_layers.png")
    fig_path_pdf = os.path.join(log_path, "figs/", f"perf_v_layers.pdf")
    fig_path_png = os.path.join(log_path, "figs/", f"perf_v_layers.png")

    plt.savefig(fig_path_pdf, pad_inches=0.01)
    plt.savefig(fig_path_png, pad_inches=0.01)
