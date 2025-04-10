import pandas as pd
import numpy as np
from src.transient_localization.utils.helpers import get_closest_index


def transform_data(filepath, min_target_freq=150):
    df = pd.read_csv(filepath, index_col=[0, 1, 2, 3], low_memory=False)

    df.index = df.index.set_levels([
        df.index.levels[0],
        df.index.levels[1],
        pd.to_numeric(df.index.levels[2], errors='coerce'),
        df.index.levels[3]
    ], level=[0, 1, 2, 3])

    available_freqs = np.sort(
        df.index.get_level_values("freq").dropna().unique().astype(float)
    )
    if available_freqs[-1] < min_target_freq:
        return None
    target_freqs = list(range(50, min_target_freq + 1, 50))
    target_buses = [6, 22, 28]

    harmonic_arrays = []
    for target in target_freqs:
        closest = get_closest_index(target, available_freqs)
        mask = df.index.get_level_values("freq").astype(float) == closest
        df_freq = df[mask]

        select_freq = df_freq[["v1", "vangle1"]]
        select_elements = select_freq.xs(0, level="element_type")
        bus_data = select_elements.groupby(level='bus').first()
        bus_data = bus_data.reindex(target_buses)

        bus_data["vangle1"] /= 180.0
        bus_data["v1"] /= 230.0

        harmonic_array = bus_data.to_numpy()
        harmonic_arrays.append(harmonic_array)

    harmonic_data = np.stack(harmonic_arrays, axis=0)
    return harmonic_data
