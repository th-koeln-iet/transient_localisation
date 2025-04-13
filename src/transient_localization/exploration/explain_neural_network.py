import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.set_style("whitegrid")
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def visualize_first_dense_layer(model, feature_names=None, layer_name=None, layer_index=None, save_figures=False, dpi=300):
    target_layer = None
    if layer_name is not None:
        target_layer = model.get_layer(layer_name)
    elif layer_index is not None:
        target_layer = model.layers[layer_index]
    else:
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                target_layer = layer
                break
    if target_layer is None:
        raise ValueError("No Dense layer found in the model.")
    weights, biases = target_layer.get_weights()
    if feature_names is None:
        feature_names = [str(i) for i in range(weights.shape[0])]
    elif len(feature_names) != weights.shape[0]:
        raise ValueError("Length of feature_names must match the number of input features.")

    aggregated_importance = np.sum(np.abs(weights), axis=1)

    sci_palette = sns.color_palette("muted")

    # ------------------------------
    # Plot 1: Horizontal Bar Plot for Aggregated Importance
    # ------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 8), dpi=dpi)
    ax1.barh(np.arange(len(aggregated_importance)), aggregated_importance, color=sci_palette[0])
    ax1.set_xlabel("Importance Score", fontsize=16)
    ax1.set_ylabel("Input Feature", fontsize=16)
    ax1.set_yticks(np.arange(len(feature_names)))
    ax1.set_yticklabels(feature_names, fontsize=14)
    ax1.set_title("Aggregated Feature Importance (Horizontal Bar Plot)", fontsize=18)
    plt.tight_layout()
    if save_figures:
        plt.savefig("aggregated_importance_barplot.png", format="png")
    else:
        plt.show()

    # ------------------------------
    # Plot 2: Heatmap for the weight matrix
    # ------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 8), dpi=dpi)
    max_abs = np.abs(weights).max()
    cmap = "coolwarm"
    sns.heatmap(weights,
                cmap=cmap,
                center=0,
                vmin=-max_abs,
                vmax=max_abs,
                square=True,
                linewidths=0.5,
                linecolor='grey',
                cbar_kws={'label': "Weight Value"},
                yticklabels=feature_names,
                ax=ax2)
    ax2.set_xlabel("Neurons in Dense Layer", fontsize=16)
    ax2.set_ylabel("Input Features", fontsize=16)
    ax2.set_title("Weight Matrix Heatmap", fontsize=18)
    ax2.tick_params(labelsize=14)
    plt.tight_layout()
    if save_figures:
        plt.savefig("weight_matrix_heatmap.png", format="png")
    else:
        plt.show()

    # ------------------------------
    # Data Aggregation and Composite Bar Plot
    # ------------------------------
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": aggregated_importance
    })

    try:
        df[['harmonic', 'measurement_device', 'signal']] = df['feature'].str.split('-', expand=True)
    except ValueError:
        raise ValueError("Feature names must be in the format 'harmonic-measurement_device-signal'.")

    agg_md = df.groupby("measurement_device", as_index=False)["importance"].sum()
    agg_md["Category"] = agg_md["measurement_device"]
    agg_md["Type"] = "Measurement Device"

    agg_signal = df.groupby("signal", as_index=False)["importance"].sum()
    agg_signal["Category"] = agg_signal["signal"]
    agg_signal["Type"] = "Signal"

    agg_harmonic = df.groupby("harmonic", as_index=False)["importance"].sum()
    agg_harmonic["Category"] = agg_harmonic["harmonic"]
    agg_harmonic["Type"] = "Harmonic Order"

    combined = pd.concat([agg_md[["Category", "importance", "Type"]],
                          agg_signal[["Category", "importance", "Type"]],
                          agg_harmonic[["Category", "importance", "Type"]]],
                         ignore_index=True)

    fig3, ax3 = plt.subplots(figsize=(10, 6), dpi=dpi)
    sns.barplot(data=combined, x="Category", y="importance", hue="Type",
                palette=sns.color_palette("muted", 3), ax=ax3)
    ax3.set_title("Aggregated Feature Importance by Category", fontsize=18)
    ax3.set_xlabel("Category", fontsize=16)
    ax3.set_ylabel("Total Importance", fontsize=16)
    ax3.tick_params(labelsize=14)
    plt.legend(title="Type", fontsize=12, title_fontsize=14)
    plt.tight_layout()
    if save_figures:
        plt.savefig("aggregated_feature_importance_by_category.png", format="png")
    else:
        plt.show()


if __name__ == '__main__':
    from src.transient_localization.models.ortho_l1_model import build_model

    name = "model_ort_l1_250Hz_0_-1"
    model = build_model(12, 45)
    model.summary()
    model.load_weights(f"{name}.h5")

    harmonics = {0: "50Hz", -1: "250Hz"}
    measurement_devices = ["MD1", "MD2", "MD3"]
    signals = ["Voltage", "Angle"]

    feature_names = []
    for harmonic_idx in [0, -1]:
        label = harmonics[harmonic_idx]
        for md in measurement_devices:
            for sig in signals:
                feature_names.append(f"{label}-{md}-{sig}")

    # Set save_figures=True to save the figures as PNG files; or set to False to display them interactively
    visualize_first_dense_layer(model, feature_names=feature_names, layer_name="dense", layer_index=0, save_figures=True)
