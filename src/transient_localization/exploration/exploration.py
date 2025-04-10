import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif


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
                    # Label is 1 if this bus is the target; note: use mapping.
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

            # Compute mutual information using sklearn (requires at least two unique labels)
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
    sns.set(style="whitegrid")
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


if __name__ == "__main__":
    from src.transient_localization.data_handling.dataset import load_dataset

    x_train, y_train = load_dataset()

    # Compute the metrics.
    results_df = compute_metrics_from_aggregated(x_train, y_train, freqs=list(range(50, 751, 50)),
                                                 target_buses=[6, 22, 28])
    print(results_df)

    # Plot the results.
    plot_metrics(results_df)
