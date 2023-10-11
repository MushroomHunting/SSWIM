class QMC_KWARG:
    PERM = "PERM"  # Permutation (for generalised halton)


class SCRAMBLING:
    GENERALISED = "GENERALISED"
    OWEN17 = "OWEN17"


class SEQUENCE:
    HALTON = "HALTON"
    SOBOL = "SOBOL"
    MC = "MC"
    R2 = "R2"


class STANDARD_KERNEL:
    """
    Analytically defined kernels (includes BIAS and LINEAR)
    """
    RBF = "RBF"
    M12 = "M12"
    M32 = "M32"
    M52 = "M52"
    LINEAR = "LINEAR"
    BIAS = "BIAS"


class METRIC:
    FROBENIUS_NORMALISED = "FROBENIUS_NORMALISED"
