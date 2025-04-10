import numpy as np


def get_closest_index(target_index, index):
    index_values = np.array(index)
    return index_values[np.abs(index_values - target_index).argmin()]


def calculate_harmonic_diff(harmonic):
    harmonic_diff = harmonic - harmonic.mean()
    harmonic_diff /= harmonic_diff.max()
    return abs(harmonic_diff)
