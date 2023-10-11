import numpy as np
import seaborn as sbn
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from torch.nn.functional import softplus

matplotlib.rcParams.update(
    {'font.size': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42,
     'legend.fontsize': 8, 'image.cmap': "viridis"})
sbn.set(font_scale=0.6)
sbn.set_context(rc={'lines.markeredgewidth': 0.25})
sbn.set_style("whitegrid")

from rfflib.features import fourier as ff37
from rfflib.utils.colour import named_CB as CBCOLS
from rfflib.utils.opt import Param
from rfflib.utils.metrics import mnlp, rmse
from rfflib.utils import datasets

from tqdm import trange
import fns
import torch

use_gpu = True
dtype = "f32"
# dtype = "f64"
gpu_id = 0  # titan v
# gpu_id = 1  # gtx 1070
# torch.cuda.set_device(gpu_id)

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
from sklearn.model_selection import train_test_split
from mpl_toolkits import mplot3d
import pandas as pd

import numpy as np
from configs.sswim_xgh2_configs import get_config
import os
import pandas as pd
# torch.set_default_tensor_type(torch.DoubleTensor)
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
    #
    # 10: "protein_structure",
    # 11: "blog_feedback",
    # 12: "buzz",
    13: "song",
}

kernel_type = "m32"
# kernel_type = "rbf"
# kernel_type = "sm"


if __name__ == "__main__":
    N_repeats = 10
    for key in all_datasets.keys():
        dataset_name = all_datasets[key]
        for experiment_id in trange(N_repeats):
            try:
                """---------------------------------   Pre-init  --------------------------------------------|
                |------------------------------------------------------------------------------------------"""
                # dataset_name = all_datasets[1]
                cfg = get_config(dataset_name)
                test_size = cfg["test_size"]

                training_iterations = cfg["optim_epochs"]

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

                # dtype = torch.float64
                X_trn = torch.tensor(X_trn.astype(dtype_np)).contiguous().cuda()
                Y_trn = torch.tensor(Y_trn.astype(dtype_np).flatten()).contiguous().cuda()
                X_tst = torch.tensor(X_tst.astype(dtype_np)).contiguous().cuda()
                Y_tst = torch.tensor(Y_tst.astype(dtype_np).flatten()).contiguous().cuda()


                # data_dim = X_trn.size(-1)

                class LargeFeatureExtractor(torch.nn.Sequential):
                    def __init__(self):
                        super(LargeFeatureExtractor, self).__init__()
                        self.add_module('linear1', torch.nn.Linear(D, 1000))
                        self.add_module('relu1', torch.nn.ReLU())
                        self.add_module('linear2', torch.nn.Linear(1000, 500))
                        self.add_module('relu2', torch.nn.ReLU())
                        self.add_module('linear3', torch.nn.Linear(500, 50))
                        self.add_module('relu3', torch.nn.ReLU())
                        self.add_module('linear4', torch.nn.Linear(50, 2))


                feature_extractor = LargeFeatureExtractor().cuda()


                class GPRegressionModel(gpytorch.models.ExactGP):
                    def __init__(self, train_x, train_y, likelihood):
                        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
                        self.mean_module = gpytorch.means.ConstantMean()

                        if kernel_type == "m32":
                            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                                gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2)),
                                num_dims=2, grid_size=100
                            )
                        elif kernel_type == "rbf":
                            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2, eps=1e-5)),
                                num_dims=2, grid_size=100
                            )
                        elif kernel_type == "sm":
                            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                                gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=5, ard_num_dims=2)),
                                num_dims=2, grid_size=100
                            )

                        self.feature_extractor = feature_extractor

                    def forward(self, x):
                        # We're first putting our data through a deep net (feature extractor)
                        # We're also scaling the features so that they're nice values
                        projected_x = self.feature_extractor(x)
                        projected_x = projected_x - projected_x.min(0)[0]
                        projected_x = 2 * (projected_x / projected_x.max(0)[0]) - 1

                        mean_x = self.mean_module(projected_x)
                        covar_x = self.covar_module(projected_x)
                        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


                likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
                model = GPRegressionModel(X_trn, Y_trn, likelihood).cuda()

                # Find optimal model hyperparameters
                model.train()
                likelihood.train()

                # Use the adam optimizer
                optimizer = torch.optim.Adam([
                    {'params': model.feature_extractor.parameters()},
                    {'params': model.covar_module.parameters()},
                    {'params': model.mean_module.parameters()},
                    {'params': model.likelihood.parameters()},
                ], lr=0.001)

                # "Loss" for GPs - the marginal log likelihood
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


                def train():
                    for i in trange(training_iterations):
                        # Zero backprop gradients
                        optimizer.zero_grad()
                        # Get output from model
                        output = model(X_trn)
                        # Calc loss and backprop derivatives
                        loss = -mll(output, Y_trn)
                        loss.backward()
                        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
                        optimizer.step()
                        if i > 50:
                            """
                            # Comment this out/in if you want to see the Test set RMSE during training
                            """
                            model.eval()
                            likelihood.eval()
                            with torch.no_grad(), gpytorch.settings.use_toeplitz(True), gpytorch.settings.fast_pred_var():
                                # preds = model(X_tst)

                                # preds = model.likelihood(model(X_tst))
                                # mus.append(preds.mean)
                                # variances.append(preds.variance)
                                preds = likelihood(model(X_tst))

                            y_mean = preds.mean.data.cpu().numpy()
                            y_var = preds.variance.data.cpu().numpy()

                            Y_tst_np = Y_tst.data.cpu().numpy().reshape(-1, 1)
                            Y_predmean_tst = y_mean.reshape(-1, 1)
                            Y_predvar_tst = y_var.reshape(-1, 1)
                            predict_metrics_mgp = {"rmse": rmse(y_actual=Y_tst_np, y_pred=y_mean.reshape(-1, 1)),  # }
                                                   "mnll": mnlp(actual_mean=Y_tst.data.cpu().numpy().reshape(-1, 1),
                                                                pred_mean=Y_predmean_tst,
                                                                pred_var=Y_predvar_tst)}
                            # print("dataset: {}".format(dataset_name))
                            print(f"[NORM SCALE]  |  <RMSE>: {predict_metrics_mgp['rmse']}    |   <MNLL>: {predict_metrics_mgp['mnll']}")

                            if Y_scaler:
                                test_rmse = rmse(y_actual=Y_scaler.inverse_transform(Y_tst_np),
                                                 y_pred=Y_scaler.inverse_transform(Y_predmean_tst))
                                test_mnll = mnlp(actual_mean=Y_scaler.inverse_transform(Y_tst_np),
                                                 pred_mean=Y_scaler.inverse_transform(Y_predmean_tst),
                                                 pred_var=Y_scaler.var_ * Y_predvar_tst)
                                print(f"[ORIG SCALE] <RMSE>: {test_rmse}, <MNLP>: {test_mnll}")

                            # print("[DKL]  |  <RMSE>: {:.5}   ".format(predict_metrics_mgp["rmse"]))  # ,
                            # predict_metrics_mgp["mnll"]))

                            model.train()  # Set to training mode again
                            likelihood.train()  # Set to training moe again
                            """
                            # Comment this out/in if you want to see the Test set RMSE during training
                            """


                # See dkl_mnist.ipynb for explanation of this flag
                with gpytorch.settings.use_toeplitz(True):
                    train()

                # Evaluation mode
                model.eval()
                likelihood.eval()
                with torch.no_grad(), gpytorch.settings.use_toeplitz(True), gpytorch.settings.fast_pred_var():
                    preds = likelihood(model(X_tst))

                y_mean = preds.mean.data.cpu().numpy()
                y_var = preds.variance.data.cpu().numpy()

                Y_tst_np = Y_tst.data.cpu().numpy().reshape(-1, 1)
                Y_predmean_tst = y_mean.reshape(-1, 1)
                Y_predvar_tst = y_var.reshape(-1, 1)

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

                result_filepath = os.path.join(log_path, f"{dataset_name}_dkl_{kernel_type}_{cfg['optim_epochs']}.csv")

                df = pd.DataFrame(results_list, columns=column_names)
                with open(result_filepath, mode="a") as f:
                    df.to_csv(f, mode="a", index=False, header=f.tell() == 0)

            except Exception as the_exception:
                print(the_exception)

                # p1 = np.sum([i.flatten().shape for i in model.feature_extractor.parameters()])
                # p2 = np.sum([i.flatten().shape for i in model.covar_module.parameters()])
                # p3 = np.sum([i.flatten().shape for i in model.mean_module.parameters()])
                # p4 = np.sum([i.flatten().shape for i in model.likelihood.parameters()])
                # print("total params: ", p1 + p2 + p3 + p4)
