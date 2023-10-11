from rfflib.utils.math_pytorch import smw_inv_precomp, logdet_correction_precomp, \
    nlml_smw, nlml_chol_fast, nlml_chol_full
from rfflib.utils.containers import Prediction
from rfflib.utils.inference import batch_generator
from rfflib.utils.opt import Param, ParamsList
from rfflib.utils import math_numpy
from rfflib.utils.cholesky import cholesky_solve, psd_safe_cholesky as cholesky
from tqdm import trange
import numpy as np
from time import time
import nlopt
import torch
from torch.nn.functional import softplus
from rfflib.utils.metrics import mnlp as mnlp_np, rmse
# from rfflib.utils.math_pytorch import mn
import matplotlib.pyplot as plt
from rfflib.utils.math_pytorch import mnlp as mnlp


class BayLinRegUICholFast:
    """
    Augmented optimizer to do approximate leave-p-out hyperparameter optimization with NLL as score metric
    Analytic Bayesian Linear Regression with SGD based NLML hyperparameter optimimsation
    Using Cholesky after the entire batch
    Also capable of doing prediction under uncertain inputs
    """

    def __init__(self,
                 kphi,
                 has_pseudo_training=False,
                 warpmul_models=None,
                 warpadd_models=None,
                 alpha=1.0,
                 beta=1.0,
                 x_trn=None,
                 y_trn=None,
                 verbose=0):
        self.x_trn = x_trn  # Mean of the gaussian inputs
        self.y_trn = y_trn
        # self.x_var = x_var ??
        self.verbose = verbose

        self.kphi = kphi
        self.M = self.kphi.M  # Feature dimensionality
        self.has_pseudo_training = has_pseudo_training
        self.warpmul_models = warpmul_models
        self.warpadd_models = warpadd_models
        self.alpha = alpha
        self.beta = beta
        self.params = [self.alpha,
                       self.beta]
        if self.has_pseudo_training is True:
            # self.params = self.params + [self.x_trn, self.y_trn]
            self.params = self.params + [self.x_trn, self.y_trn]

        self.params_list = ParamsList(self.get_params(),
                                      numpy_2_torch=False,
                                      torch_2_numpy=False,
                                      keep_tensor_list=True)

        self.optim_class = None  # Reserved for pytorch sgd based methods like adam
        self.optimizer = None  # Reserved for pytorch sgd based methods like adam
        self.logging = {"loss": []}

    def get_params(self,
                   full_depth=True):
        """
        Returns all parameter objects from this class
        Optionally (default) returns every parameter from all parameter holding objects contained within
        :param full_depth:
        :return:
        """

        if full_depth is True:
            params = self.params + self.kphi.get_params()
            if self.warpmul_models is not None:
                for warpmul_model in self.warpmul_models:
                    params += warpmul_model.get_params()
            if self.warpadd_models is not None:
                for warpadd_model in self.warpadd_models:
                    params += warpadd_model.get_params()
            return params
        else:
            return self.params

    def fit(self,
            x_trn=None,
            x_var=None,  # for uncertain inputs prediction
            y_trn=None,
            with_grad=False):
        """
        Fit the weights
        :return:
        """
        if x_trn is None:
            x_trn = self.x_trn
            y_trn = self.y_trn
        if self.has_pseudo_training is True:
            x_trn = x_trn.forward()
            y_trn = y_trn.forward()
        if self.warpadd_models is not None:
            # t1 = time()
            for j in range(len(self.warpmul_models)):
                warpmul_model = self.warpmul_models[j]
                warpadd_model = self.warpadd_models[j]
                warpmul_model.fit(with_grad=with_grad)
                warpadd_model.fit(with_grad=with_grad)
            # t2 = time()
            # print(f"time taken for fitting : {t2 - t1} s")
            # Do the warping!
            for j in range(len(self.warpadd_models)):
                warpmul_model = self.warpmul_models[j]
                warpadd_model = self.warpadd_models[j]

                warpmul_pred = warpmul_model.predict(x_trn,
                                                     x_var=x_var,  # The variance of the previous step!
                                                     with_grad=with_grad,
                                                     with_var=True)
                warpadd_pred = warpadd_model.predict(x_trn,
                                                     x_var=x_var,  # The variance of the previous step!
                                                     with_grad=with_grad,
                                                     with_var=True)

                warpmul_mean = warpmul_pred.mean
                warpmul_var = warpmul_pred.var
                warpadd_mean = warpadd_pred.mean
                warpadd_var = warpadd_pred.var
                """ Update the variance term """
                if x_var is None:
                    x_var = warpmul_var * (x_trn * x_trn) + warpadd_var
                else:
                    # Do the V[XY] = V[X]V[Y] + V[X]E[Y]^2 + V[Y]E[X]^2 calculation
                    # Note, predvar is a (N,1) column vector, and x_trn is (N,D) matrix. this broadcasts.
                    EX = warpmul_mean
                    EY = x_trn
                    VX = warpmul_var
                    VY = x_var
                    # x_var = (VX * VY + VX * EY ** 2 + VY * EX ** 2) * (x_trn * x_trn) + warpadd_var
                    x_var = (VX * VY + VX * EY ** 2 + VY * EX ** 2) + warpadd_var
                """ Update the mean term """
                x_trn = x_trn * warpmul_mean + warpadd_mean

        self.kphi.sample_frequencies()

        PHI = self.kphi.transform(x_trn, X_var=x_var)
        A = torch.matmul(PHI.t(), PHI) + (self.alpha.forward() / self.beta.forward()) * torch.eye(self.kphi.M)
        self.R_lower = cholesky(A, upper=False)
        b = torch.matmul(PHI.t(), y_trn)
        self.Rb_solve, _ = torch.solve(b, self.R_lower)
        self.mu, _ = torch.solve(self.Rb_solve, self.R_lower.t())

    def fit_get_loss(self,
                     x_trn=None,
                     x_var=None,
                     y_trn=None,
                     loss_type="nlml",
                     log_the_loss=False,
                     with_grad=False):
        """
        Run a fit as well as calculate the loss (NLML)
        :param log_the_loss:
        :return:
        """
        if x_trn is None:
            x_trn = self.x_trn
            y_trn = self.y_trn

        if loss_type == "nlml":
            self.fit(x_trn=x_trn,
                     x_var=x_var,
                     y_trn=y_trn,
                     with_grad=with_grad)
            loss = nlml_chol_fast(y_trn=y_trn,
                                  r=self.R_lower,
                                  rb_solve=self.Rb_solve,
                                  n=x_trn.shape[0],
                                  m=self.M,
                                  alpha=self.alpha.forward(),
                                  beta=self.beta.forward())

        ''' optimize '''

        if log_the_loss is True:
            self.logging["loss"].append(loss.item())

        return loss  # if you want to read this out, just call .item() or .data.numpy()

    def optimize(self,
                 optimizer=None,
                 optimizer_kwargs=None,
                 loss_type="nlml",  # nlml, mnlp
                 N_f=3,  # Number of splits
                 N_ftrn=3,  # Number of samples per split
                 N_fval=3,  # Number of validation
                 log_the_loss=True,
                 test_logging=False):

        """
        Optimize the hyperparameters of this object
        :param optimizer:
        :param optimizer_kwargs:
        :return:
        """
        if optimizer == "adam":
            self.optim_class = torch.optim.Adam
            # self.optim_class = RAdam
            # self.optim_class = Ranger
            if "optim_epochs" in optimizer_kwargs:
                epochs = optimizer_kwargs["optim_epochs"]
            else:
                epochs = 100
            if "optim_kwargs" in optimizer_kwargs:
                optim_kwargs = optimizer_kwargs["optim_kwargs"]
            else:
                optim_kwargs = {"lr": .075,
                                "betas": (0.9, 0.999),
                                "eps": 1e-10}
            self.optimizer = self.optim_class(self.params_list.params_flat, **optim_kwargs)

            if test_logging:
                self.logging["rmse_test_normscale"] = []
                self.logging["mnlp_test_normscale"] = []
                self.logging["rmse_test_origscale"] = []
                self.logging["mnlp_test_origscale"] = []
                self.logging["nlml"] = []
                X_tst = test_logging["X_tst"]
                Y_tst_np = test_logging["Y_tst_np"]
                Y_scaler = test_logging["Y_scaler"]

            if loss_type == "mnlp":
                N_trn = self.x_trn.shape[0]
                fold_trn_idxs = []
                fold_val_idxs = []
                all_possible_trn_idxs = []
                for _ in range(N_f):
                    idxs = np.arange(N_trn)
                    np.random.shuffle(idxs)
                    fold_val_idxs.append(idxs[0:N_fval])
                    fold_trn_idxs.append(idxs[N_fval:N_fval + N_ftrn])
                    all_possible_trn_idxs.append(np.copy(idxs[N_fval:]))

            for i in trange(epochs, position=0, leave=True):
                self.optimizer.zero_grad()
                ''' Fit our models (And save the NLML)'''

                if loss_type == "nlml":
                    loss = self.fit_get_loss(loss_type=loss_type,
                                             with_grad=True)

                # tic = time()
                loss.backward(retain_graph=False)
                # toc = time()
                if i < epochs - 1:
                    self.optimizer.step()
                    print("LOSS: {}".format(loss.item()))
                else:
                    print("FINAL LOSS: {}".format(loss.item()))

                if log_the_loss:
                    self.logging["loss"].append(loss.item())

                if test_logging:
                    if loss_type == "mnlp":
                        self.fit()
                    prediction_tst = self.predict(x=X_tst, with_var=True, with_grad=False)
                    Y_predmean_tst = prediction_tst.mean.cpu().data.numpy()
                    Y_predvar_tst = prediction_tst.var.cpu().data.numpy()
                    # print(f"\nalpha: {self.alpha.forward().cpu().data.numpy().round(4)}, beta: {self.beta.forward().cpu().data.numpy().round(4)}, ", )

                    if Y_scaler:
                        test_rmse = rmse(y_actual=Y_scaler.inverse_transform(Y_tst_np),
                                         y_pred=Y_scaler.inverse_transform(Y_predmean_tst))
                        test_mnll = mnlp_np(actual_mean=Y_scaler.inverse_transform(Y_tst_np),
                                            pred_mean=Y_scaler.inverse_transform(Y_predmean_tst),
                                            pred_var=Y_scaler.var_ * Y_predvar_tst)
                        self.logging["rmse_test_origscale"].append(test_rmse)
                        self.logging["mnlp_test_origscale"].append(test_mnll)
                        print(f"[ORIG SCALE] <RMSE>: {test_rmse}, <MNLP>: {test_mnll}", end="")
                    test_rmse = rmse(y_actual=Y_tst_np,
                                     y_pred=Y_predmean_tst)
                    test_mnll = mnlp_np(actual_mean=Y_tst_np,
                                        pred_mean=Y_predmean_tst,
                                        pred_var=Y_predvar_tst)
                    self.logging["rmse_test_normscale"].append(test_rmse)
                    self.logging["mnlp_test_normscale"].append(test_mnll)
                    print(f"[NORM SCALE] <RMSE>: {test_rmse}, <MNLP>: {test_mnll}")



        elif optimizer == "nlopt":
            if "nlopt_maxevals" in optimizer_kwargs:
                nlopt_maxevals = optimizer_kwargs["nlopt_maxevals"]
            else:
                nlopt_maxevals = 42
            if "nlopt_optimizer" in optimizer_kwargs:
                nlopt_optimizer = optimizer_kwargs["nlopt_optimizer"]
            else:
                nlopt_optimizer = nlopt.GN_DIRECT
            # nlopt_optimizer = nlopt.GN_ISRES
            # nlopt_optimizer = nlopt.GN_DIRECT_NOSCAL
            if "nlopt_verbose" in optimizer_kwargs:
                nlopt_verbose = optimizer_kwargs["nlopt_verbose"]
            else:
                nlopt_verbose = 1

            if nlopt_maxevals == 0:
                self.fit()
                return

            def loss_nlopt(x,
                           grad=None,
                           loss_fn=self.fit_get_loss,
                           params_list=self.params_list,
                           verbose=nlopt_verbose):
                params_list.set_params(x)  # 1. take the x and XBO.params_list.set()
                loss = loss_fn()  # 2. XBLR.fit_get_loss()
                if verbose >= 2:
                    print("loss: ", loss)
                return loss

            D_opt = self.params_list.params_flat.shape[0]
            opt = nlopt.opt(nlopt_optimizer, D_opt)
            opt.set_lower_bounds(self.params_list.gmins)
            opt.set_upper_bounds(self.params_list.gmaxs)
            opt.set_min_objective(loss_nlopt)
            opt.set_maxeval(nlopt_maxevals)
            params_flat_optimized = opt.optimize(self.params_list.params_inits)
            best_loss = opt.last_optimum_value()
            if nlopt_verbose >= 1:
                print("optimum at {}".format(params_flat_optimized.round(3)))
                print("minimum value {:.6f} ".format(best_loss))
            # print("result code = ", opt.last_optimize_result())

            # Set the optimized parameters
            self.params_list.set_params(new_params=params_flat_optimized)
            # Fit the model again with these
            self.fit()
        else:
            opt = optimizer
            params_flat_optimized = opt.optimize(self.params_list.params_inits)
            best_loss = opt.last_optimum_value()
            self.params_list.set_params(new_params=params_flat_optimized)
            self.fit()

    def predict(self,
                x,
                x_var=None,
                with_var=True,
                with_grad=False):
        """

        :param x:
        :param with_var:
        :param with_grad:
        :return:
        """

        prediction = Prediction()  # Initialise a prediction result object to dump results into

        if self.warpmul_models is not None:
            # Do the warping!
            for j in range(len(self.warpadd_models)):
                warpmul_model = self.warpmul_models[j]
                warpadd_model = self.warpadd_models[j]

                warpmul_pred = warpmul_model.predict(x,
                                                     x_var=x_var,  # The variance of the previous step!
                                                     with_grad=with_grad,
                                                     with_var=True)
                warpadd_pred = warpadd_model.predict(x,
                                                     x_var=x_var,  # The variance of the previous step!
                                                     with_grad=with_grad,
                                                     with_var=True)

                warpmul_mean = warpmul_pred.mean
                warpmul_var = warpmul_pred.var
                warpadd_mean = warpadd_pred.mean
                warpadd_var = warpadd_pred.var

                """ Update the variance term """
                if x_var is None:
                    # Do the DETERMINISTIC inputs warping  X_det * Y_unc
                    x_var = warpmul_var * (x * x) + warpadd_var
                else:
                    # Do the V[XY] = V[X]V[Y] + V[X]E[Y]^2 + V[Y]E[X]^2 calculation
                    # Note, predvar is a (N,1) column vector, and x_trn is (N,D) matrix. this broadcasts.
                    # x_var = warpmul_var * (x * x) + warpadd_var
                    EX = warpmul_mean
                    EY = x
                    VX = warpmul_var
                    VY = x_var
                    # x_var = (VX * VY + VX * EY ** 2 + VY * EX ** 2) * (x * x) + warpadd_var
                    x_var = (VX * VY + VX * EY ** 2 + VY * EX ** 2) + warpadd_var
                """ Update the mean term """
                x = x * warpmul_mean + warpadd_mean
        if with_grad:
            PHI = self.kphi.transform(x, X_var=x_var)
            Y_preds = torch.matmul(PHI, self.mu)
            # This is for the multiple point predictive means
            prediction.mean = Y_preds  # , dim=1, keepdim=True)
        else:
            with torch.no_grad():
                PHI = self.kphi.transform(x, X_var=x_var)
                Y_preds = torch.matmul(PHI, self.mu)
                # This is for the multiple point predictive means
                prediction.mean = Y_preds  # , dim=1, keepdim=True)

        # print("Got pred mean for {} points in {} s".format(prediction.mean.shape,np.round(time_taken, 5)))
        if with_var:
            RPhiPred_solve, _ = torch.solve(PHI.t(), self.R_lower)
            prediction.var = (1 / self.beta.forward() + (1 / self.beta.forward()) * torch.norm(RPhiPred_solve, dim=0) ** 2).t()
        return prediction
