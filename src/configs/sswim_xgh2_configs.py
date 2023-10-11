import numpy as np
from rfflib.utils.enums import STANDARD_KERNEL, SEQUENCE


def get_config(dataset_name, N_warplayers=None, ):
    config = {}

    # Default settings
    config["test_size"] = 1.0 / 3.0
    config["N_latent"] = 10000
    if N_warplayers is None:
        config["N_warplayers"] = 2  # Standard RFF. No warping!
    else:
        config["N_warplayers"] = N_warplayers
    config["M_blr"] = int(512)
    config["M_warpmul"] = int(512)
    config["M_warpadd"] = int(512)
    config["prescale_div"] = 0.005
    config["ls_top_div"] = 1.0
    config["alpha_initval"] = 0.5
    config["alpha_warpmul_initval"] = 0.1
    config["alpha_warpadd_initval"] = 0.1
    config["beta_initval"] = 1.5
    config["beta_warpmul_initval"] = 1
    config["beta_warpadd_initval"] = 1
    config["Y_warpmul_init_std"] = np.sqrt(1 / 0.1)
    config["Y_warpadd_init_std"] = np.sqrt(1 / 0.1)

    config["kernel_type"] = STANDARD_KERNEL.M32
    config["kernel_type_warpmul"] = STANDARD_KERNEL.M32
    config["kernel_type_warpadd"] = STANDARD_KERNEL.M32
    config["sequence_type"] = SEQUENCE.MC
    config["sequence_type_warpmul"] = SEQUENCE.MC
    config["sequence_type_warpadd"] = SEQUENCE.MC
    config["optim_kwargs"] = {"lr": 0.05, "betas": (0.1, 0.1)}
    config["optim_epochs"] = 150
    """------------------------------------------------
                                                        """
    if dataset_name == "airfoil_noise":
        config["prescale_div"] = 0.1
        config["ls_top_div"] = 0.1
        """
        ------------------------------------------------"""

        """------------------------------------------------
                                                        """
    elif dataset_name == "concrete_compressive":
        config["prescale_div"] = 0.001
        config["ls_top_div"] = 10
    #     """
    #     ------------------------------------------------"""
    #     """------------------------------------------------
    #                                                     """
    elif dataset_name == "parkinsons_total":
        """
        ------------------------------------------------"""
        config["prescale_div"] = 0.1
        config["ls_top_div"] = 10
        """------------------------------------------------
                                                        """
    elif dataset_name == "bike_sharing_hourly":
        """
        ------------------------------------------------"""
        config["prescale_div"] = 0.001
        config["ls_top_div"] = 0.1
        """------------------------------------------------
                                                        """
    # if dataset_name == "ct_slice":
    #     """
    #     ------------------------------------------------"""
    #     config["M_blr"] = int(256)
    #     config["M_warpmul"] = int(256)
    #     config["M_warpadd"] = int(256)
    #     """------------------------------------------------
    #                                                     """
    if dataset_name == "superconductivity":
        """
        ------------------------------------------------"""
        config["M_blr"] = int(1024)
        config["M_warpmul"] = int(1024)
        config["M_warpadd"] = int(1024)
        config["prescale_div"] = 0.1
        config["ls_top_div"] = 0.1
        """------------------------------------------------
                                                        """
    elif dataset_name == "abalone":
        """
        ------------------------------------------------"""
        # config["test_size"] = 1.0 / 3.0
        config["test_size"] = 1 - (1000.5 / 4177)  # To match WGP settings
        config["kernel_type"] = STANDARD_KERNEL.RBF
        config["kernel_type_warpmul"] = STANDARD_KERNEL.RBF
        config["kernel_type_warpadd"] = STANDARD_KERNEL.RBF
        config["alpha_initval"] = 0.3
        config["beta_initval"] = 3.0
        config["prescale_div"] = 0.001
        config["ls_top_div"] = 0.01
        config["optim_kwargs"] = {"lr": 0.03}
        """------------------------------------------------
                                                        """
    elif dataset_name == "creep":
        """
        ------------------------------------------------"""
        # config["test_size"] = 1.0 / 3.0
        config["test_size"] = 1 - (800 / 2066)  # To match WGP settings
        config["kernel_type"] = STANDARD_KERNEL.RBF
        config["kernel_type_warpmul"] = STANDARD_KERNEL.RBF
        config["kernel_type_warpadd"] = STANDARD_KERNEL.RBF
        config["optim_kwargs"] = {"lr": 0.03}
        """------------------------------------------------
                                                        """
    elif dataset_name == "ailerons":
        """
        ------------------------------------------------"""
        # # config["test_size"] = 1.0 / 3.0
        config["test_size"] = 1 - (1003 / 7174)  # To match WGP settings
        config["kernel_type"] = STANDARD_KERNEL.RBF
        config["kernel_type_warpmul"] = STANDARD_KERNEL.RBF
        config["kernel_type_warpadd"] = STANDARD_KERNEL.RBF
        config["alpha_initval"] = 0.3
        config["beta_initval"] = 1.0
        config["prescale_div"] = 0.0001
        config["ls_top_div"] = 1
        config["optim_kwargs"] = {"lr": 0.003}

        """------------------------------------------------
                                                        """
    elif dataset_name == "elevators":
        """
        ------------------------------------------------"""
        config["prescale_div"] = 0.001
        config["ls_top_div"] = 0.1

        """------------------------------------------------
                                                        """
    elif dataset_name == "protein_structure":
        """
        ------------------------------------------------"""
        config["M_blr"] = int(1024)
        config["M_warpmul"] = int(1024)
        config["M_warpadd"] = int(1024)
        config["prescale_div"] = 1
        """------------------------------------------------
                                                        """
    elif dataset_name == "buzz":
        """
        ------------------------------------------------"""
        config["M_blr"] = int(128)
        config["M_warpmul"] = int(128)
        config["M_warpadd"] = int(128)

        """------------------------------------------------
                                                        """
    elif dataset_name == "song":
        """
        ------------------------------------------------"""
        config["M_blr"] = int(256)
        config["M_warpmul"] = int(256)
        config["M_warpadd"] = int(256)
        config["ls_top_div"] = 10

    return config
