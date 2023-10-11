import numpy as np
import torch


class Param:
    def __init__(self,
                 init,
                 forward_fn=None,
                 gmin=None,
                 gmax=None,
                 name=None,
                 requires_grad=False):
        """
        NOTE: This does not enforce the min max! it is up to the optimizer to read the
              min and max values and enforce the constraint
              I.e. when we call forward, it just returns the value
        :param init:
        :param forward_fn:
        :param min_global:
        :param max_global:
        :param requires_grad:
        """
        self.value = torch.tensor(init).detach().requires_grad_(requires_grad)
        self.gmin = gmin
        self.gmax = gmax
        self.name = None
        if forward_fn is None:
            self.forward_fn = self.get_value
        else:
            self.forward_fn = forward_fn

    def get_value(self, x=None):
        return self.value

    def forward(self):
        return self.forward_fn(self.value)


class ParamsList:
    def __init__(self,
                 params_list,
                 numpy_2_torch=True,
                 torch_2_numpy=True,
                 keep_tensor_list=False):
        self.params_list = params_list
        self.numpy_2_torch = numpy_2_torch
        self.torch_2_numpy = torch_2_numpy
        self.keep_tensor_list = keep_tensor_list
        self.params_flat = None
        self.start_idxs = None
        self.end_idxs = None
        self.params_inits = None
        self.gmins = None
        self.gmaxs = None

        self.extract_params(keep_tensor_list=True)

    def extract_params(self, keep_tensor_list=False):
        """
        Extracts a flattened vector representation of all the individual parameters
        :param do_flatten:
        :return:
        """
        start_idxs = []
        end_idxs = []
        params_flat = []
        params_inits = []
        gmins = []
        gmaxs = []
        for i, param_obj in enumerate(self.params_list):
            param_flat = param_obj.value  # .view(-1)
            param_init = param_obj.value  # .view(-1)
            if param_obj.gmin is not None:
                gmin = param_obj.gmin.value
                gmins.append(gmin)
            if param_obj.gmax is not None:
                gmax = param_obj.gmax.value
                gmaxs.append(gmax)
            if self.torch_2_numpy is True:
                # param_flat = param_flat.flatten
                param_flat = param_flat.data.cpu().numpy()
            params_flat.append(param_flat)
            params_inits.append(param_init)

            param_flat_len = param_flat.shape[0]
            if i == 0:
                start_idxs.append(0)
                end_idxs.append(start_idxs[0] + param_flat_len)
            else:
                start_idxs.append(end_idxs[i - 1])
                end_idxs.append(start_idxs[i] + param_flat_len)

        if keep_tensor_list is True:
            self.params_flat = params_flat
            self.params_inits = params_inits
            self.gmins = gmins
            self.gmaxs = gmaxs
        else:
            if self.torch_2_numpy is True:
                self.params_flat = np.concatenate(params_flat)
                self.params_inits = np.concatenate(params_inits)
                if self.gmins is not None:
                    self.gmins = np.concatenate(gmins)
                if self.gmaxs is not None:
                    self.gmaxs = np.concatenate(gmaxs)
            else:
                self.params_flat = torch.cat(params_flat)
                self.params_inits = torch.cat(params_inits)
                if self.gmins is not None:
                    self.gmins = torch.cat(gmins)
                if self.gmaxs is not None:
                    self.gmaxs = torch.cat(gmaxs)
        self.start_idxs = start_idxs
        self.end_idxs = end_idxs

    def set_params(self, new_params):
        for i, param_obj in enumerate(self.params_list):
            param_obj.value = new_params[self.start_idxs[i]:self.end_idxs[i]].reshape(param_obj.value.shape)
            if self.numpy_2_torch is True:
                param_obj.value = torch.tensor(param_obj.value)
