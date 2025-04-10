import os
import json
from tqdm import tqdm
import numpy as np
from src.transient_localization.data_handling.transform import transform_data
from src.transient_localization.utils.name_parser import NameParser
from src.transient_localization import config as cfg


def create_dataset(min_target_freq=150):
    sim_folders = os.listdir(cfg.filepath_sim_data)
    calculated = os.listdir(f"./train_data_{min_target_freq}Hz/")
    skip_counter = 0
    for folder in tqdm(sim_folders):
        if folder + ".json" in calculated:
            continue
        dataset = {"x": [], "y": []}
        folder_path = os.path.join(cfg.filepath_sim_data, folder)
        sim_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        for sim_file in sim_files:
            parser = NameParser(sim_file)
            filepath = os.path.join(folder_path, sim_file)
            x = transform_data(filepath, min_target_freq)
            if x is None:
                skip_counter += 1
                print(f"Skipping {sim_file} because it doesn't cover frequencies up to {min_target_freq}Hz. Skipped {skip_counter} folders.")
                break
            y = parser.bus
            dataset["x"].append(x.tolist())
            dataset["y"].append(y)
        if x is not None:
            with open(f"./train_data_{min_target_freq}Hz/{folder}.json", 'w') as json_file:
                json.dump(dataset, json_file, indent=4)


def load_dataset(path="./train_data_350Hz/"):
    x_complete, y_complete = [], []
    files = os.listdir(path)
    for file in files:
        with open(f"{path}{file}", 'r') as json_file:
            data = json.load(json_file)
        x = np.stack(data["x"])
        y = np.stack(data["y"])
        x_complete.append(x)
        y_complete.append(y)
    x_train = np.vstack(x_complete)
    y_train = np.hstack(y_complete)
    return x_train, y_train


if __name__ == '__main__':
    create_dataset(min_target_freq=250)
