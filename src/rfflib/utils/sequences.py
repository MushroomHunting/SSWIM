import numpy as np
import ghalton as gh
from rfflib.utils.enums import SCRAMBLING, SEQUENCE, QMC_KWARG
import torch.distributions as tdists
import torch


class Sequence(object):
    def __init__(self,
                 N,
                 D,
                 seed=42,
                 sequence_type=SEQUENCE.HALTON,
                 scramble_type=SCRAMBLING.OWEN17,
                 kwargs={QMC_KWARG.PERM: None},
                 ):
        self.N = N
        self.D = D
        self.seed = seed
        self.sequence_type = sequence_type
        self.scramble_type = scramble_type
        self.kwargs = kwargs
        self.sequencer = None
        self.points = None
        self.init_sequencer()
        self.init_points()

    def init_sequencer(self):
        # ---------------------------------------#
        #                Halton                  #
        # ---------------------------------------#
        if self.sequence_type == SEQUENCE.HALTON:
            if self.scramble_type == SCRAMBLING.OWEN17:
                pass
            elif self.scramble_type == SCRAMBLING.GENERALISED:
                if self.kwargs[QMC_KWARG.PERM] is None:
                    perm = gh.EA_PERMS[:self.D]  # Default permutation
                else:
                    perm = self.kwargs[QMC_KWARG.PERM]
                self.sequencer = gh.GeneralizedHalton(perm)
            else:
                self.sequencer = gh.Halton(int(self.D))
        # ---------------------------------------#
        #                  R2                    #
        # ---------------------------------------#
        elif self.sequence_type == SEQUENCE.R2:
            pass

        # ---------------------------------------#
        #              Monte-Carlo               #
        # ---------------------------------------#
        elif self.sequence_type == SEQUENCE.MC:
            self.sequencer = tdists.Uniform(torch.tensor(0.0),
                                            torch.tensor(1.0))

    def init_points(self):
        if self.sequence_type == SEQUENCE.MC:
            self.points = self.sequencer.sample(sample_shape=(self.N, self.D))
        elif self.sequence_type == SEQUENCE.R2:
            pass
        elif self.sequence_type == SEQUENCE.HALTON and self.scramble_type == SCRAMBLING.OWEN17:
            self.points = tf.Variable(initial_value=tfp.mcmc.sample_halton_sequence(dim=self.D,
                                                                                    num_results=self.N,
                                                                                    randomized=True,
                                                                                    seed=self.seed),
                                      dtype=self.tfdt,
                                      trainable=False)

            self.points = torch.tensor(np.array(self.sequencer.get(int(self.N))))

    def resample_points(self):
        if self.sequence_type == SEQUENCE.MC:
            self.points = self.sequencer.sample(sample_shape=(self.N, self.D))
        else:  # Low Discrepancy
            pass

