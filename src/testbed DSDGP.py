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

gpu_id = 0  # titan v
# gpu_id = 1  # gtx 1070
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
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
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
    #
    # 10: "protein_structure",
    # 11: "blog_feedback",
    # 12: "buzz",
    13: "song",
}

kernel_type = "m32"
# kernel_type = "rbf"

num_output_dims = 10
num_epochs = 150  # 300
num_samples = 30
learning_rate = 0.01
N_repeats = 6

if __name__ == "__main__":
    for key in all_datasets.keys():
        dataset_name = all_datasets[key]
        for experiment_id in trange(N_repeats):
            try:
                cfg = get_config(dataset_name)
                test_size = cfg["test_size"]
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

                train_dataset = TensorDataset(X_trn, Y_trn)
                train_loader = DataLoader(train_dataset, batch_size=1024 * 1, shuffle=True)


                class ToyDeepGPHiddenLayer(DeepGPLayer):
                    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
                        if output_dims is None:
                            inducing_points = torch.randn(num_inducing, input_dims)
                            batch_shape = torch.Size([])
                        else:
                            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
                            batch_shape = torch.Size([output_dims])

                        variational_distribution = CholeskyVariationalDistribution(
                            num_inducing_points=num_inducing,
                            batch_shape=batch_shape
                        )

                        variational_strategy = VariationalStrategy(
                            self,
                            inducing_points,
                            variational_distribution,
                            learn_inducing_locations=True
                        )

                        super(ToyDeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

                        if mean_type == 'constant':
                            self.mean_module = ConstantMean(batch_shape=batch_shape)
                        else:
                            self.mean_module = LinearMean(input_dims)

                        if kernel_type == "rbf":
                            self.covar_module = ScaleKernel(
                                RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
                                batch_shape=batch_shape, ard_num_dims=None
                            )
                        elif kernel_type == "m32":
                            self.covar_module = ScaleKernel(
                                MaternKernel(nu=1.5, batch_shape=batch_shape, ard_num_dims=input_dims),
                                batch_shape=batch_shape, ard_num_dims=None
                            )
                        self.linear_layer = Linear(input_dims, 1)

                    def forward(self, x):
                        mean_x = self.mean_module(x)  # self.linear_layer(x).squeeze(-1)
                        covar_x = self.covar_module(x)
                        return MultivariateNormal(mean_x, covar_x)

                    def __call__(self, x, *other_inputs, **kwargs):
                        """
                        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
                        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
                        hidden layer's outputs and the input data to hidden_layer2.
                        """
                        if len(other_inputs):
                            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                                x = x.rsample()

                            processed_inputs = [
                                inp.unsqueeze(0).expand(self.num_samples, *inp.shape)
                                for inp in other_inputs
                            ]

                            x = torch.cat([x] + processed_inputs, dim=-1)

                        return super().__call__(x, are_samples=bool(len(other_inputs)))


                class DeepGP(DeepGPModel):
                    def __init__(self, train_x_shape):
                        hidden_layer = ToyDeepGPHiddenLayer(
                            input_dims=train_x_shape[-1],
                            output_dims=num_output_dims,
                            mean_type='linear',
                        )

                        last_layer = ToyDeepGPHiddenLayer(
                            input_dims=hidden_layer.output_dims,
                            output_dims=None,
                            mean_type='constant',
                        )

                        super().__init__()

                        self.hidden_layer = hidden_layer
                        self.last_layer = last_layer
                        self.likelihood = GaussianLikelihood()

                    def forward(self, inputs):
                        hidden_rep1 = self.hidden_layer(inputs)
                        output = self.last_layer(hidden_rep1)
                        return output

                    def predict(self, test_loader):
                        with torch.no_grad():
                            mus = []
                            variances = []
                            lls = []
                            for x_batch, y_batch in test_loader:
                                preds = model.likelihood(model(x_batch))
                                mus.append(preds.mean)
                                variances.append(preds.variance)
                                lls.append(model.likelihood.log_marginal(y_batch, model(x_batch)))

                        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)


                # model = DeepGP(X_trn.shape).cuda()
                model = DeepGP(X_trn.shape)
                if torch.cuda.is_available():
                    model = model.cuda()

                optimizer = torch.optim.Adam([
                    {'params': model.parameters()},
                ], lr=learning_rate)
                mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, X_trn.shape[-2]))

                for i in trange(num_epochs):
                    # Within each iteration, we will go over each minibatch of data
                    # minibatch_iter = tqdm.tqdm_notebook(train_loader, desc="Minibatch", leave=False)
                    # for x_batch, y_batch in minibatch_iter:
                    for minibatch_i, (x_batch, y_batch) in enumerate(train_loader):
                        with gpytorch.settings.num_likelihood_samples(num_samples):
                            optimizer.zero_grad()
                            output = model(x_batch)
                            loss = -mll(output, y_batch)
                            loss.backward()
                            optimizer.step()

                            # minibatch_iter.set_postfix(loss=loss.item())

                    # """
                    # # Comment this out/in if you want to see the Test set RMSE during training
                    # """
                    # test_dataset = TensorDataset(X_tst, Y_tst)
                    # test_loader = DataLoader(test_dataset, batch_size=1024)
                    #
                    # model.eval()
                    # predictive_means, predictive_variances, test_lls = model.predict(test_loader)
                    #
                    # y_mean = predictive_means.mean(0).data.cpu().numpy()
                    # y_var = predictive_variances.mean(0).data.cpu().numpy()
                    #
                    # Y_tst_np = Y_tst.data.cpu().numpy().reshape(-1, 1)
                    # Y_predmean_tst = y_mean.reshape(-1, 1)
                    # Y_predvar_tst = y_var.reshape(-1, 1)
                    #
                    # results_dict = {}
                    # if do_Y_standardisation:
                    #     results_dict["RMSE_orig"] = rmse(y_actual=Y_scaler.inverse_transform(Y_tst_np),
                    #                                      y_pred=Y_scaler.inverse_transform(Y_predmean_tst))
                    #     results_dict["MNLP_orig"] = mnlp(actual_mean=Y_scaler.inverse_transform(Y_tst_np),
                    #                                      pred_mean=Y_scaler.inverse_transform(Y_predmean_tst),
                    #                                      pred_var=Y_scaler.var_ * Y_predvar_tst)
                    #
                    # results_dict["RMSE_norm"] = rmse(y_actual=Y_tst_np, y_pred=Y_predmean_tst)
                    # results_dict["MNLP_norm"] = mnlp(actual_mean=Y_tst_np, pred_mean=Y_predmean_tst,
                    #                                  pred_var=Y_predvar_tst)
                    # column_names = ["MSE_orig",
                    #                 "RMSE_orig",
                    #                 "MNLP_orig",
                    #                 "MSE_norm",
                    #                 "RMSE_norm",
                    #                 "MNLP_norm"]
                    # results_list = [[results_dict["RMSE_orig"] ** 2,
                    #                  results_dict["RMSE_orig"],
                    #                  results_dict["MNLP_orig"],
                    #                  results_dict["RMSE_norm"] ** 2,
                    #                  results_dict["RMSE_norm"],
                    #                  results_dict["MNLP_norm"], ]]
                    # print("\n[DSDGP]  |  <RMSE_norm>: {:.5}    |   <MNLP_norm>: {:.5}".format(results_dict["RMSE_norm"],
                    #                                                                           results_dict["MNLP_norm"]))
                    #
                    # model.train()
                    # """
                    # # Comment this out/in if you want to see the Test set RMSE during training
                    # """
                test_dataset = TensorDataset(X_tst, Y_tst)
                test_loader = DataLoader(test_dataset, batch_size=1024 * 1)

                model.eval()
                predictive_means, predictive_variances, test_lls = model.predict(test_loader)

                y_mean = predictive_means.mean(dim=0).data.cpu().numpy()
                y_var = predictive_variances.mean(dim=0).data.cpu().numpy()

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
                print("\n[DSDGP]  |  <RMSE>: {:.5}    |   <MNLP>: {:.5}".format(results_dict["RMSE_orig"],
                                                                                results_dict["MNLP_orig"]))

                result_filepath = os.path.join(log_path, f"{dataset_name}_dsdgp_{kernel_type}_{cfg['optim_epochs']}.csv")

                df = pd.DataFrame(results_list, columns=column_names)
                with open(result_filepath, mode="a") as f:
                    df.to_csv(f, mode="a", index=False, header=f.tell() == 0)
                # optimizer = torch.optim.Adam([
                #     {'params': model.parameters()},
                # ], lr=0.01)
                # mll = VariationalELBO(model.likelihood, model, X_trn.shape[-2])
                #
                # import time
                #
                # with gpytorch.settings.fast_computations(log_prob=False, solves=False):
                #     for i in trange(num_epochs):
                #         for minibatch_i, (x_batch, y_batch) in enumerate(train_loader):
                #             start_time = time.time()
                #             optimizer.zero_grad()
                #
                #             output = model(x_batch)
                #             loss = -mll(output, y_batch)
                #             print('Epoch %d [%d/%d] - Loss: %.3f - - Time: %.3f' % (i + 1, minibatch_i, len(train_loader), loss.item(), time.time() - start_time))
                #
                #             loss.backward()
                #             optimizer.step()
                #
                #         """
                #         # Comment this out/in if you want to see the Test set RMSE during training
                #         """
                #         model.eval()
                #         with torch.no_grad():
                #             if X_tst.shape[0] < 400:
                #                 predictive_means, predictive_variances = model.predict(X_tst)
                #                 y_mean = predictive_means.mean(0).data.cpu().numpy()
                #                 y_var = predictive_variances.mean(0).data.cpu().numpy()
                #             else:
                #                 predictive_means, predictive_variances = model.predict(X_tst[0:500, :])
                #                 y_mean_1 = predictive_means.mean(0).data.cpu().numpy()
                #                 y_var_1 = predictive_variances.mean(0).data.cpu().numpy()
                #
                #                 predictive_means, predictive_variances = model.predict(X_tst[500:, :])
                #                 y_mean_2 = predictive_means.mean(0).data.cpu().numpy()
                #                 y_var_2 = predictive_variances.mean(0).data.cpu().numpy()
                #
                #                 y_mean = np.concatenate([y_mean_1, y_mean_2])
                #                 y_var = np.concatenate([y_var_1, y_var_2])
                #
                #             # rmse = torch.mean(torch.pow(predictive_means.mean(0) - Y_tst)).sqrt()
                #             # with torch.no_grad():
                #             #     test_ll = torch.distributions.Normal(predictive_means, predictive_variances.sqrt()).log_prob(Y_tst.logsumexp(dim=0) - math.log(predictive_means.size(0)))
                #
                #             # print(rmse)
                #             # print(test_ll.mean())
                #
                #             predict_metrics_mgp = {"rmse": rmse(y_actual=Y_tst.data.cpu().numpy().reshape(-1, 1), y_pred=y_mean.reshape(-1, 1)),  # }
                #                                    "mnll": mnlp(actual_mean=Y_tst.data.cpu().numpy().reshape(-1, 1), pred_mean=y_mean.reshape(-1, 1),
                #                                                 pred_var=y_var.reshape(-1, 1))}
                #             # print("dataset: {}".format(dataset_name))
                #             print("[DKL]  |  <RMSE>: {:.5}    |   <MNLL>: {:.5}".format(predict_metrics_mgp["rmse"],
                #                                                                         predict_metrics_mgp["mnll"]))
                #             # print("[DSDGP]  |  <RMSE>: {:.5}   ".format(predict_metrics_mgp["rmse"]))  # ,
                #             # # # predict_metrics_mgp["mnll"]))
                #
                #         model.train()  # Set to training mode again
            except Exception as the_exception:
                print(the_exception)
