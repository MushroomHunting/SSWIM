import numpy as np
from rfflib.utils.enums import STANDARD_KERNEL, SEQUENCE


def get_config(dataset_name):
    config = {}
    # Default settings
    config["test_size"] = 1.0 / 3.0
    return config
