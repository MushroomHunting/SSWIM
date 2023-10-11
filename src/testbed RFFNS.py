import numpy as np
import torch
import os

use_gpu = True
# use_gpu = False
# dtype = "f32"
dtype = "f64"
gpu_id = 1  # titan v
# gpu_id = 1 # gtx 1070
torch.cuda.set_device(gpu_id)

if dtype == "f32":
    dtype_np = np.float32
    if use_gpu:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
elif dtype == "f64":
    dtype_np = np.float64
    if use_gpu:
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)

import seaborn as sbn
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import rfflib.utils.math_pytorch as mathtorch
import rfflib.utils.math_numpy as mathnp

matplotlib.rcParams.update(
    {'font.size': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42,
     'legend.fontsize': 8, 'image.cmap': "viridis"})
sbn.set(font_scale=0.6)
sbn.set_context(rc={'lines.markeredgewidth': 0.25})
sbn.set_style("whitegrid")

from tqdm import trange
from rfflib.models.LinearRegression import \
    BayLinRegUICholFast as BLRUI
from rfflib.features import fourier as ff37
from rfflib.utils.colour import named_CB as CBCOLS
from rfflib.utils.opt import Param
from rfflib.utils.metrics import mnlp, rmse
from rfflib.utils import dataloader
import pandas as pd
import torch
from configs.sswim_xgh2_configs import get_config
from pathlib import Path

home = str(Path.home())  # logging path
log_path = f"{home}/experiment_results/"

all_datasets = {
    0: "elevators",
    1: "airfoil_noise",
    2: "concrete_compressive",
    3: "parkinsons_total",
    4: "bike_sharing_hourly",
    5: "ct_slice",
    6: "superconductivity",
    # 7: "abalone",
    # 8: "creep",
    # 9: "ailerons",

    10: "protein_structure",
    # 11: "blog_feedback",
    12: "buzz",
    13: "song",
}

if __name__ == "__main__":
    N_warplayers = 0
    N_repeats = 10
    do_surface_plot = False
    for key in all_datasets.keys():
        dataset_name = all_datasets[key]
        for experiment_id in trange(N_repeats):
            try:
                """---------------------------------   Pre-init  --------------------------------------------|
                |------------------------------------------------------------------------------------------"""
                cfg = get_config(dataset_name, N_warplayers=N_warplayers)

                # dataset_name ="abalone" #(D=8)
                # dataset_name = "creep"
                # dataset_name ="ailerons"

                test_size = cfg["test_size"]

                # test_size = 1 - (1000.5 / 4177 )# abalone
                # test_size = 1 - (800 / 2066)  # creep
                # test_size = 1 - (1003 / 7174 ) # ailerons

                shuffle = True
                do_X_standardisation = True
                do_Y_standardisation = True

                X_trn_np, \
                X_tst_np, \
                Y_trn_np, \
                Y_tst_np, \
                X_scaler, \
                Y_scaler = dataloader.get_data(dataset_name,
                                               test_size=test_size,
                                               shuffle=True,
                                               standardize_x=do_X_standardisation,
                                               standardize_y=do_Y_standardisation)

                N_trn, D = X_trn_np.shape
                N_tst, _ = X_tst_np.shape
                print(f"\nD: {D}")
                print(f"N_trn: {N_trn}, N_tst: {N_tst}")

                X_trn = torch.tensor(X_trn_np.astype(dtype_np))
                X_tst = torch.tensor(X_tst_np.astype(dtype_np))
                Y_trn = torch.tensor(Y_trn_np.astype(dtype_np))
                Y_tst = torch.tensor(Y_tst_np.astype(dtype_np))

                N_latent = cfg["N_latent"]

                warpmul_models = []
                warpadd_models = []

                # For initialising each layer
                X_trn_mean_jm1 = X_trn_np
                X_trn_var_jm1 = None

                min_eps = 1e-5
                # X_trn_range_jm1 = np.clip(X_trn_range_jm1, a_min=min_eps, a_max=np.inf)
                """ BLR initialisations """
                M_blr = cfg["M_blr"]

                pc_lower = 0
                pc_upper = 100 - pc_lower
                pc1, pc2 = np.percentile(X_trn_mean_jm1, q=[pc_lower, pc_upper], axis=0, keepdims=True)
                pc_diff = pc2 - pc1
                prescale_div = max(cfg["prescale_div"] * 10, 2)
                ls_top_div = 1.0
                ls_prescale = np.array(pc_diff) * np.sqrt(D / prescale_div)
                ls_prescale[ls_prescale == 0] = 1 / 1e-5
                alpha_initval = cfg["alpha_initval"]
                beta_initval = cfg["beta_initval"]
                ns_type = "lebesgue_stieltjes"
                kernel_type = cfg["kernel_type"]
                sequence_type = cfg["sequence_type"]
                scramble_type = None
                scramble_type_warpmul = None
                scramble_type_warpadd = None

                requires_grad_ui = True
                requires_grad_warpmul = True
                requires_grad_warpadd = True
                is_uncertain_inputs = True

                all_idxs = np.arange(N_trn)

                """---------------------------------   INITS: TOP level   -----------------------------------|
                |------------------------------------------------------------------------------------------"""
                ls_init = mathnp.softplus2_inv(ls_prescale / ls_top_div)
                meanshift_init = (0.0 * ls_prescale)  # initialise to zero meanshift...
                alpha_init = mathnp.softplus2_inv(alpha_initval * np.ones((1, 1)))
                beta_init = mathnp.softplus2_inv(beta_initval * np.ones((1, 1)))
                """---------------------------------   UIBLR: TOP prediction level (predictor) --------------|
                |------------------------------------------------------------------------------------------"""
                # ls = Param(init=ls_init.astype(dtype_np), forward_fn=mathtorch.softplus2, requires_grad=requires_grad_ui)
                ls = [Param(init=ls_init.astype(dtype_np), forward_fn=mathtorch.softplus2, requires_grad=requires_grad_ui),
                      Param(init=ls_init.astype(dtype_np), forward_fn=mathtorch.softplus2, requires_grad=requires_grad_ui)]
                meanshift = [Param(init=meanshift_init.astype(dtype_np), requires_grad=requires_grad_ui),
                             Param(init=meanshift_init.astype(dtype_np), requires_grad=requires_grad_ui)]
                KPHIUI = ff37.LengthscaleKernel(M=M_blr, D=D, ls=ls, ns_type=ns_type, meanshift=meanshift,
                                                kernel_type=kernel_type,
                                                sequence_type=sequence_type, scramble_type=scramble_type)

                alpha = Param(init=alpha_init.astype(dtype_np),
                              forward_fn=mathtorch.softplus2, requires_grad=requires_grad_ui)
                beta = Param(init=beta_init.astype(dtype_np), forward_fn=mathtorch.softplus2,
                             requires_grad=requires_grad_ui)
                BLR_ui = BLRUI(kphi=KPHIUI,
                               alpha=alpha,
                               beta=beta,
                               has_pseudo_training=False,  # This is the top level predicti
                               warpmul_models=warpmul_models,
                               warpadd_models=warpadd_models,
                               x_trn=X_trn,
                               y_trn=Y_trn)

                print("Training RFF Non Stationary model...", end="")
                # test_logging = None
                test_logging = {"X_tst": X_tst,
                                "Y_tst_np": Y_tst_np,
                                "Y_scaler": Y_scaler
                                }
                optim_kwargs = cfg["optim_kwargs"]
                optim_epochs = cfg["optim_epochs"]
                loss_type = "nlml"
                print("done!")

                optimizer_kwargs = {"optim_kwargs": optim_kwargs,
                                    "optim_epochs": optim_epochs}
                BLR_ui.optimize(optimizer="adam",
                                loss_type=loss_type,
                                optimizer_kwargs=optimizer_kwargs,
                                test_logging=test_logging)

                prediction_tst = BLR_ui.predict(x=X_tst, with_var=True, with_grad=False)
                Y_predmean_tst = prediction_tst.mean.cpu().data.numpy()
                Y_predvar_tst = prediction_tst.var.cpu().data.numpy()

                results_dict = {}
                if do_Y_standardisation:
                    results_dict["RMSE_orig"] = rmse(y_actual=Y_scaler.inverse_transform(Y_tst_np),
                                                     y_pred=Y_scaler.inverse_transform(Y_predmean_tst))
                    results_dict["MNLP_orig"] = mnlp(actual_mean=Y_scaler.inverse_transform(Y_tst_np),
                                                     pred_mean=Y_scaler.inverse_transform(Y_predmean_tst),
                                                     pred_var=Y_scaler.var_ * Y_predvar_tst)

                results_dict["RMSE_norm"] = rmse(y_actual=Y_tst_np, y_pred=Y_predmean_tst)
                results_dict["MNLP_norm"] = mnlp(actual_mean=Y_tst_np, pred_mean=Y_predmean_tst,
                                                 pred_var=Y_predvar_tst)
                column_names = ["MSE_orig",
                                "RMSE_orig",
                                "MNLP_orig",
                                "MSE_norm",
                                "RMSE_norm",
                                "MNLP_norm"]
                results_list = [[results_dict["RMSE_orig"] ** 2,
                                 results_dict["RMSE_orig"],
                                 results_dict["MNLP_orig"],
                                 results_dict["RMSE_norm"] ** 2,
                                 results_dict["RMSE_norm"],
                                 results_dict["MNLP_norm"], ]]
                print("[SSWIM]  |  <RMSE>: {:.5}    |   <MNLP>: {:.5}".format(results_dict["RMSE_orig"],
                                                                              results_dict["MNLP_orig"]))

                result_filepath = os.path.join(log_path, f"{dataset_name}_rffns_{cfg['optim_epochs']}.csv")

                df = pd.DataFrame(results_list, columns=column_names)
                with open(result_filepath, mode="a") as f:
                    df.to_csv(f, mode="a", index=False, header=f.tell() == 0)

            except Exception as the_exception:
                print(the_exception)
