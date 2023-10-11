from time import time
import numpy as np
import seaborn as sbn
import matplotlib

import torch

matplotlib.rcParams.update(
    {'font.size': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42,
     'legend.fontsize': 8, 'image.cmap': "viridis"})
sbn.set(font_scale=0.6)
sbn.set_context(rc={'lines.markeredgewidth': 0.25})
sbn.set_style("whitegrid")

from rfflib.utils.metrics import mnlp, rmse

from tqdm import trange
import torch

use_gpu = True
dtype = "f32"
# dtype = "f64"

# gpu_id = 0  # titan v
gpu_id = 1  # gtx 1070
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

torch.cuda.set_device(gpu_id)

import pandas as pd
import os
import numpy as np

# %set_env CUDA_VISIBLE_DEVICES=0
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Linear
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel, InducingPointKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood

# from gpytorch.models.deep_gps import AbstractDeepGPLayer, AbstractDeepGP, DeepLikelihood
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP as DeepGPModel, DeepLikelihood
from gpytorch.mlls import DeepApproximateMLL

# torch.set_default_tensor_type(torch.DoubleTensor)
from configs.sswim_xgh2_configs import get_config
from rfflib.utils import dataloader
import gpytorch

from pathlib import Path

home = str(Path.home())  # logging path
log_path = f"{home}/experiment_results/"

all_datasets = {
    # 0: "elevators",
    # 1: "airfoil_noise",
    # 2: "concrete_compressive",
    # 3: "parkinsons_total",
    # 4: "bike_sharing_hourly",
    # 5: "ct_slice",
    # 6: "superconductivity",
    # 10: "protein_structure",
    12: "buzz",
    13: "song",
}

kernel_type = "m32"
# kernel_type = "rbf"

N_repeats = 10

if __name__ == "__main__":
    for key in all_datasets.keys():
        dataset_name = all_datasets[key]
        for experiment_id in trange(N_repeats):
            try:
                cfg = get_config(dataset_name)
                test_size = cfg["test_size"]
                num_epochs = cfg['optim_epochs']
                shuffle = True
                do_X_standardisation = True
                do_Y_standardisation = True

                X_trn, \
                X_tst, \
                Y_trn, \
                Y_tst, \
                X_scaler, \
                Y_scaler = dataloader.get_data(dataset_name,
                                               test_size=test_size,
                                               shuffle=True,
                                               standardize_x=do_X_standardisation,
                                               standardize_y=do_Y_standardisation)
                N_trn, D = X_trn.shape

                # dtype = torch.float32
                X_trn = torch.tensor(X_trn.astype(dtype_np)).contiguous().cuda()
                Y_trn = torch.tensor(Y_trn.astype(dtype_np).flatten()).contiguous().cuda()
                X_tst = torch.tensor(X_tst.astype(dtype_np)).contiguous().cuda()
                Y_tst = torch.tensor(Y_tst.astype(dtype_np).flatten()).contiguous().cuda()

                from torch.utils.data import TensorDataset, DataLoader

                train_dataset = TensorDataset(X_trn, Y_trn)
                train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

                test_dataset = TensorDataset(X_tst, Y_tst)
                test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

                from gpytorch.models import ApproximateGP
                from gpytorch.variational import CholeskyVariationalDistribution
                from gpytorch.variational import VariationalStrategy


                class GPModel(ApproximateGP):
                    def __init__(self, inducing_points):
                        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
                        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
                        super(GPModel, self).__init__(variational_strategy)
                        self.mean_module = gpytorch.means.ConstantMean()
                        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

                        if kernel_type == "rbf":
                            self.covar_module = ScaleKernel(
                                RBFKernel(  # batch_shape=batch_shape,
                                    ard_num_dims=X_trn.shape[1]),
                                # batch_shape=batch_shape,
                                ard_num_dims=None
                            )
                        elif kernel_type == "m32":
                            self.covar_module = ScaleKernel(
                                MaternKernel(nu=1.5,
                                             # batch_shape=batch_shape,
                                             ard_num_dims=X_trn.shape[1]),
                                # batch_shape=batch_shape,
                                ard_num_dims=None
                            )

                    def forward(self, x):
                        mean_x = self.mean_module(x)
                        covar_x = self.covar_module(x)
                        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


                inducing_points = X_trn[:512, :]
                model = GPModel(inducing_points=inducing_points)
                likelihood = gpytorch.likelihoods.GaussianLikelihood()

                if torch.cuda.is_available():
                    model = model.cuda()
                    likelihood = likelihood.cuda()

                model.train()
                likelihood.train()

                # We use SGD here, rather than Adam. Emperically, we find that SGD is better for variational regression
                optimizer = torch.optim.Adam([
                    {'params': model.parameters()},
                    {'params': likelihood.parameters()},
                ], lr=0.1)

                # Our loss object. We're using the VariationalELBO
                mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_trn.size(0))

                for i in trange(num_epochs):
                    # Within each iteration, we will go over each minibatch of data
                    # minibatch_iter = tqdm.tqdm_notebook(train_loader, desc="Minibatch", leave=False)
                    # for x_batch, y_batch in minibatch_iter:
                    for minibatch_i, (x_batch, y_batch) in enumerate(train_loader):
                        optimizer.zero_grad()
                        output = model(x_batch)
                        loss = -mll(output, y_batch)
                        loss.backward()
                        optimizer.step()

                """
                # Evaluation mode
                """
                model.eval()
                likelihood.eval()
                Y_predmean_tst = None  # np.array([[]])
                Y_predvar_tst = None  # np.array([[]])
                with torch.no_grad(), gpytorch.settings.use_toeplitz(True), gpytorch.settings.fast_pred_var():
                    # preds = likelihood(model(X_tst))
                    # preds = likelihood(model(test_loader))
                    for i_test, (x_batch_test, y_batch_test) in enumerate(test_loader):
                        preds = likelihood(model(x_batch_test))

                        y_mean = preds.mean.data.cpu().numpy()
                        y_var = preds.variance.data.cpu().numpy()
                        if i_test == 0:
                            Y_predmean_tst = y_mean.reshape(-1, 1)
                            Y_predvar_tst = y_var.reshape(-1, 1)
                        else:
                            Y_predmean_tst = np.concatenate([Y_predvar_tst, y_mean.reshape(-1, 1)], axis=0)
                            Y_predvar_tst = np.concatenate([Y_predvar_tst, y_var.reshape(-1, 1)], axis=0)

                Y_tst_np = Y_tst.data.cpu().numpy().reshape(-1, 1)

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
                print("\n[DKL]  |  <RMSE>: {:.5}    |   <MNLP>: {:.5}".format(results_dict["RMSE_orig"],
                                                                              results_dict["MNLP_orig"]))

                result_filepath = os.path.join(log_path, f"{dataset_name}_svgp_{kernel_type}_{cfg['optim_epochs']}.csv")

                df = pd.DataFrame(results_list, columns=column_names)
                with open(result_filepath, mode="a") as f:
                    df.to_csv(f, mode="a", index=False, header=f.tell() == 0)


            except Exception as the_exception:
                print(the_exception)
