import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter1d


def get_lds_kernel_window(ks, sigma):
    half_ks = (ks - 1) // 2
    base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks

    # Using Gaussian Kernel
    kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))

    return kernel_window


def prepare_weights(data, max_target=51, lds_ks=5, lds_sigma=2):
    value_dict = {x: 0 for x in range(max_target)}
    labels = data.iloc[:, -1].tolist()
    for label in labels:
        value_dict[min(max_target - 1, int(label))] += 1

    # Inverse for LDS
    value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}

    lds_kernel_window = get_lds_kernel_window(lds_ks, lds_sigma)

    smoothed_value = convolve1d(
        np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')

    num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

    weights = [np.float32(1 / x) for x in num_per_label]
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * x for x in weights]
    return weights
