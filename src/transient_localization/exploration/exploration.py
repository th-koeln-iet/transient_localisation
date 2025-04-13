import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score


sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.set_style("whitegrid")
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def compute_metrics_from_aggregated(X, y_target,
                                    freqs=list(range(50, 751, 50)),
                                    target_buses=[6, 22, 28]):
    n_samples, n_freq, n_buses, n_features = X.shape
    target_bus_to_index = {bus: idx for idx, bus in enumerate(target_buses)}

    results = []
    feature_names = ['v1', 'vangle1']

    for f_idx, freq in enumerate(freqs):
        for feat_idx, feat in enumerate(feature_names):
            measurements = []
            labels = []
            for i in range(n_samples):
                target = y_target[i]
                for j in range(n_buses):
                    val = X[i, f_idx, j, feat_idx]
                    measurements.append(val)
                    label = 1 if (target in target_bus_to_index and target_bus_to_index[target] == j) else 0
                    labels.append(label)
            measurements = np.array(measurements)
            labels = np.array(labels)

            valid_mask = ~np.isnan(measurements)
            measurements_valid = measurements[valid_mask].reshape(-1, 1)
            labels_valid = labels[valid_mask]

            target_vals = measurements_valid[labels_valid == 1].flatten()
            non_target_vals = measurements_valid[labels_valid == 0].flatten()
            var_target = np.var(target_vals) if len(target_vals) > 0 else np.nan
            var_non_target = np.var(non_target_vals) if len(non_target_vals) > 0 else np.nan
            mean_target = np.mean(target_vals) if len(target_vals) > 0 else np.nan

            if var_non_target > 0 and mean_target != 0:
                snr = 10 * np.log10((mean_target ** 2) / var_non_target)
            else:
                snr = np.nan

            if len(np.unique(labels_valid)) > 1:
                mi = mutual_info_classif(measurements_valid, labels_valid,
                                         discrete_features=False, random_state=0)[0]
            else:
                mi = np.nan

            results.append({
                "freq": freq,
                "feature": feat,
                "var_target": var_target,
                "var_non_target": var_non_target,
                "MI": mi,
                "SNR": snr
            })
    results_df = pd.DataFrame(results)
    return results_df


def plot_metrics(results_df):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    ax = axs[0]
    sns.lineplot(data=results_df, x="freq", y="var_target", hue="feature", marker="o", ax=ax)
    ax.set_title("Variance (Target Measurements) vs. Frequency")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Variance")

    ax = axs[1]
    sns.lineplot(data=results_df, x="freq", y="MI", hue="feature", marker="o", ax=ax)
    ax.set_title("Mutual Information vs. Frequency")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mutual Information")

    ax = axs[2]
    sns.lineplot(data=results_df, x="freq", y="SNR", hue="feature", marker="o", ax=ax)
    ax.set_title("SNR vs. Frequency")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("SNR (dB)")

    plt.tight_layout()
    plt.show()


def compute_joint_mi(X_pair, labels, n_bins=10):
    X_disc = np.empty_like(X_pair, dtype=int)
    for col in range(X_pair.shape[1]):
        col_vals = X_pair[:, col]
        bins = np.linspace(np.nanmin(col_vals), np.nanmax(col_vals), n_bins + 1)
        X_disc[:, col] = np.digitize(col_vals, bins)
    combined = np.array([f"{x1}_{x2}" for x1, x2 in X_disc])
    joint_mi = mutual_info_score(labels, combined)
    return joint_mi


def compute_pairwise_joint_mi(X, y_target, fixed_freq, freqs=list(range(50, 751, 50)),
                              target_buses=[6, 22, 28], feature_names=['v1', 'vangle1'], n_bins=10):
    n_samples, n_freq, n_buses, n_features = X.shape
    target_bus_to_index = {bus: idx for idx, bus in enumerate(target_buses)}
    try:
        fixed_index = freqs.index(fixed_freq)
    except ValueError:
        raise ValueError(f"Fixed frequency {fixed_freq} not in provided frequency list {freqs}")

    results = []

    for partner_freq in freqs:
        if partner_freq == fixed_freq:
            continue
        partner_index = freqs.index(partner_freq)
        for feat_idx, feat in enumerate(feature_names):
            X_pair_list = []
            labels_list = []
            for i in range(n_samples):
                target = y_target[i]
                for j in range(n_buses):
                    fixed_val = X[i, fixed_index, j, feat_idx]
                    partner_val = X[i, partner_index, j, feat_idx]
                    if np.isnan(fixed_val) or np.isnan(partner_val):
                        continue
                    X_pair_list.append([fixed_val, partner_val])
                    label = 1 if (target in target_bus_to_index and target_bus_to_index[target] == j) else 0
                    labels_list.append(label)
            if len(X_pair_list) == 0:
                joint_mi = np.nan
            else:
                X_pair_array = np.array(X_pair_list)
                labels_array = np.array(labels_list)
                joint_mi = compute_joint_mi(X_pair_array, labels_array, n_bins=n_bins)
            results.append({
                "partner_freq": partner_freq,
                "feature": feat,
                "joint_MI": joint_mi
            })
    df_pairwise = pd.DataFrame(results)
    return df_pairwise


def plot_pairwise_mi(df_pairwise, fixed_freq, dpi=300, save_fig=False, save_filename=None):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    sns.lineplot(data=df_pairwise, x="partner_freq", y="joint_MI", hue="feature", marker="o", ax=ax)
    ax.set_title(f"Joint Mutual Information: {fixed_freq}Hz with Partner Frequencies", fontsize=18)
    ax.set_xlabel("Partner Frequency (Hz)", fontsize=16)
    ax.set_ylabel("Joint MI", fontsize=16)
    plt.tight_layout()
    if save_fig:
        if save_filename is None:
            save_filename = f"pairwise_MI_{fixed_freq}Hz.png"
        plt.savefig(save_filename, format="png")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    from src.transient_localization.data_handling.dataset import load_dataset
    x_train, y_train = load_dataset()
    results_df = compute_metrics_from_aggregated(x_train, y_train, freqs=list(range(50, 751, 50)),
                                                 target_buses=[6, 22, 28])
    print(results_df)
    plot_metrics(results_df)

    df_pairwise_50 = compute_pairwise_joint_mi(x_train, y_train,
                                               fixed_freq=50,
                                               freqs=list(range(50, 751, 50)),
                                               target_buses=[6, 22, 28],
                                               feature_names=['v1', 'vangle1'],
                                               n_bins=10)
    plot_pairwise_mi(df_pairwise_50, fixed_freq=50, dpi=300,
                     save_fig=True, save_filename="pairwise_MI_50Hz.png")

    df_pairwise_700 = compute_pairwise_joint_mi(x_train, y_train,
                                                fixed_freq=700,
                                                freqs=list(range(50, 751, 50)),
                                                target_buses=[6, 22, 28],
                                                feature_names=['v1', 'vangle1'],
                                                n_bins=10)
    plot_pairwise_mi(df_pairwise_700, fixed_freq=700, dpi=300,
                     save_fig=True, save_filename="pairwise_MI_700Hz.png")
