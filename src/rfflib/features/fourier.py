from rfflib.utils.enums import SEQUENCE, SCRAMBLING, QMC_KWARG, STANDARD_KERNEL
from rfflib.features import samplers, transformers
from rfflib.utils.sequences import Sequence
from rfflib.utils import math_numpy
from rfflib.utils.opt import Param
from torch.nn.functional import softplus
import torch.nn as nn
import torch


class Linear():
    def __init__(self,
                 D,
                 has_bias_term=True,
                 transformer=transformers.linear):
        super().__init__()
        self.D = D,
        self.has_bias_term = has_bias_term
        self.__transformer = transformer
        if self.has_bias_term is True:
            self.M = self.D + 1
        else:
            self.M = self.D

    def transform(self, X):
        """
        Applies the instance's sample weight expansion function to some data X
        h is the constant magnitude scalar
        :param X: The input data we want to transform
        :return: The fourier represented feature map for this particular kernel
        """
        return self.__transformer(X, self.has_bias_term)


class LengthscaleKernel():
    """
    Fourier Features for standard shift invariant kernels
      - RBF
      - M12
      - M32
      - M52
    """

    def __init__(self,
                 M,
                 D,
                 ls=Param(init=1.0,
                          forward_fn=None,
                          gmin=0.02,
                          gmax=2.00),
                 ns_type=None,
                 meanshift=None,
                 kernel_type=STANDARD_KERNEL.RBF,
                 sequence_type=SEQUENCE.HALTON,
                 scramble_type=SCRAMBLING.OWEN17,
                 kwargs={QMC_KWARG.PERM: None}):
        # sampler=samplers.standard_kernel,
        # transformer=transformers.cos_sin):
        """
        :param M:   int
                    The dimensionality of our features
                    M = 2m because we are using the [cos(wx),sin(wx)]
        :param D:   int
                    The dimensionality of our input data
        :param h:   float
                    The constant scaling parameter of the corresponding kernel
                    This is typically written as h^2 for the full kernel
                    and so, this h is that h
        :param ls: Param()  [Optimizable]
        :param sampler: The sampling function
        :param transformer: The transformer function
        """
        super().__init__()
        if M % 2 != 0:
            raise ValueError("M must be an even number")
        self.M = M
        self.m = M // 2  # This is half the number of features
        self.D = D
        self.ns_type = ns_type
        self.kernel_type = kernel_type
        self.sequencer_type = sequence_type
        self.scramble_type = scramble_type
        self.kwargs = kwargs
        self.sequence = Sequence(N=self.m, D=self.D,
                                 sequence_type=self.sequencer_type,
                                 scramble_type=self.scramble_type,
                                 kwargs=self.kwargs)

        self.S = None
        self.ls = ls
        self.meanshift = meanshift
        self.__sampler = samplers.standard_kernel  # The sampling function. e.g. FSF_RBF
        if self.ns_type == "lebesgue_stieltjes":
            self.__transformer = transformers.cos_sin_ns
        else:
            self.__transformer = transformers.cos_sin_ui
        self.sample_frequencies()  # This is defined in the actual feature mapper (e.g. RFF_RBF())

    def get_params(self):
        if self.ns_type == "lebesgue_stieltjes":
            params = [*self.ls]
            if self.meanshift:
                params += [*self.meanshift]
            return params
        elif self.ns_type is None:
            return [self.ls]

    def sample_frequencies(self):
        """
        Allows one to resamples the internal spectral weights
        This would typically occur after an optimisation step
        The lengthscale can be optimized separately
        Assumes self.params order is known apriori
        I.e. the sampler and transformer should match each other
        :note: During optimisation, ensure that this method is called during the fitting process otherwise the parameters
               won't be updated!
        """

        if self.ns_type == "lebesgue_stieltjes":
            self.S = self.__sampler(sequence=self.sequence,
                                    kernel_type=self.kernel_type,
                                    ls=[self.ls[0].forward(), self.ls[1].forward()],
                                    meanshift=[self.meanshift[0].forward(), self.meanshift[1].forward()],
                                    ns_type=self.ns_type)
        else:
            self.S = self.__sampler(sequence=self.sequence,
                                    kernel_type=self.kernel_type,
                                    ls=self.ls.forward(),
                                    ns_type=self.ns_type)
        # self.S = self.__sampler(sequence=self.sequence, kernel_type=self.kernel_type) / self.ls.forward()

    def transform(self, X, X_var=None):
        """
        Applies the instance's sample weight expansion function to some data X
        h is the constant magnitude scalar
        :param X:       The input data we want to transform
        :param X_var:   The covariance of the X
                        NOTE: For this paper, we are dealing only with diagonal covariance
        :return: The fourier represented feature map for this particular kernel
        """
        return self.__transformer(X, X_var=X_var, S=self.S)


class BBQKernel():
    """
    Black Box Quantile Quasi-Monte-Carlo (Fourier) Features
    for approximating valid black-box kernels
    This is the upgraded version that allows:
        1. Lebesgue-Steltjes non-stationary kernels, and
        2. ARD quantiles (different quantile per dimension)
    """

    def __init__(self,
                 M,
                 D,
                 qpoints,
                 interp="pchipasymp",
                 is_ard=True,
                 ns_type=None,
                 sequence_type=SEQUENCE.HALTON,
                 scramble_type=SCRAMBLING.GENERALISED,
                 kwargs={QMC_KWARG.PERM: None},
                 uncertain_inputs=False):
        """
        :param M:   int
                    The final dimensionality of our features. M=2m
        :param D:   int
                    The dimensionality of our input data
        :param ns_type: None, "functional" or "lebesgue_steljes"
                        The nonstationary method
        :param sampler: The sampling function
        :param transformer: The transformer function
        """
        super().__init__()
        if M % 2 != 0:
            raise ValueError("M must be an even number")
        self.M = M
        self.m = M // 2  # This is half the number of features
        self.D = D
        self.qpoints = qpoints
        self.interp = interp
        self.is_ard = is_ard
        self.ns_type = ns_type
        self.sequencer_type = sequence_type
        self.scramble_type = scramble_type
        self.kwargs = kwargs
        self.uncertain_inputs = uncertain_inputs
        self.sequence = Sequence(N=self.m, D=self.D,
                                 sequence_type=self.sequencer_type,
                                 scramble_type=self.scramble_type,
                                 kwargs=self.kwargs)
        self.S = None  # The spectral weights  # possible a list of S. i.e. [S1, S2] for lebesgue_stieltjes
        self.__sampler = samplers.bbq_kernel  # The sampling function. e.g. FSF_RBF
        if self.ns_type == "lebesgue_stieltjes":
            self.__transformer = transformers.cos_sin_ns
        else:
            if self.uncertain_inputs is True:
                self.__transformer = transformers.cos_sin_ui
            else:
                self.__transformer = transformers.cos_sin

    def get_params(self):
        if self.ns_type == "lebesgue_stieltjes":
            return self.qpoints[0].get_params() + self.qpoints[1].get_params()
        elif self.ns_type is None:
            return self.qpoints.get_params()

    def sample_frequencies(self):
        """
        Allows one to resample the internal spectral weights
        This amounts to passing the uniform samples through the new quantile
        This would typically occur after an optimisation step
        The sampler and transformer should match each other
        """
        self.S = self.__sampler(self.sequence, self.qpoints, interp_type=self.interp, ns_type=self.ns_type)

    def transform(self, X, X_var=None):
        """
        Applies the instance's sample weight expansion function to some data X
        h is the constant magnitude scalar
        :param X:       The input data we want to transform
        :param X_var:   The covariance of the X
                        NOTE: For this paper, we are dealing only with diagonal covariance
        :return: The fourier represented feature map for this particular kernel
        """
        if self.uncertain_inputs is True:
            return self.__transformer(X, X_var=X_var, S=self.S)
        else:
            return self.__transformer(X, S=self.S)
