import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.transient_localization.data_handling.dataset import load_dataset
from keras import metrics
from src.transient_localization.utils.sgdr_scheduler import SGDRScheduler
from tensorflow.keras import optimizers
from src.transient_localization.models.dnn import build_model
from keras.callbacks import CSVLogger


sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.set_style("whitegrid")
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def evaluate_harmonics_for_deep_learning():
    x_t, y_train = load_dataset("./train_data_750Hz/")

    num_samples = x_t.shape[0]
    batch_size = np.ceil(float(x_t.shape[0]) / 1)

    top_k3 = metrics.SparseTopKCategoricalAccuracy(k=3, name="TopK3")
    top_k5 = metrics.SparseTopKCategoricalAccuracy(k=5, name="TopK5")

    results = {}

    for order, harmonic in enumerate(range(50, 751, 50)):
        x_train = x_t[:, order, :, :]
        x_train = x_train.reshape((num_samples, -1))
        model = build_model(x_train.shape[1], 44 + 1)

        schedule = SGDRScheduler(min_lr=1e-8,
                                 max_lr=1e-3,
                                 steps_per_epoch=np.ceil(len(x_train * 0.9) / batch_size),
                                 lr_decay=0.8,
                                 cycle_length=500,
                                 mult_factor=1.5)
        csv_logger = CSVLogger(f'./logs/pre_training_{harmonic}.log')

        model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy', top_k3, top_k5])

        hist = model.fit(x_train, y_train,
                         batch_size=int(batch_size),
                         validation_split=0.1,
                         epochs=5000,
                         callbacks=[schedule, csv_logger],
                         verbose=2)
        results[harmonic] = hist.history


def plot_training_logs(logs_dir='./logs/'):
    file_pattern = os.path.join(logs_dir, "pre_training_*.log")
    log_files = glob.glob(file_pattern)

    if not log_files:
        print(f'No log files matching the pattern {file_pattern} were found.')
        return

    logs_dict = {}
    harmonic_pattern = re.compile(r'pre_training_(\d+)\.log')

    for filepath in log_files:
        filename = os.path.basename(filepath)
        match = harmonic_pattern.search(filename)
        if not match:
            print(f"Filename {filename} does not match expected pattern. Skipping.")
            continue

        harmonic = int(match.group(1))
        try:
            df = pd.read_csv(filepath)
            logs_dict[harmonic] = df
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not logs_dict:
        print("No valid log files were loaded.")
        return

    summary_rows = []
    required_columns = ['epoch', 'val_loss', 'val_accuracy', 'val_TopK3', 'val_TopK5']
    for harmonic in sorted(logs_dict.keys()):
        df = logs_dict[harmonic]
        if not all(col in df.columns for col in required_columns):
            print(f"Warning: Log for harmonic {harmonic} is missing required columns. Skipping.")
            continue

        best_val_loss = df['val_loss'].min()
        best_val_accuracy = df['val_accuracy'].max()
        best_val_top3 = df['val_TopK3'].max()
        best_val_top5 = df['val_TopK5'].max()

        summary_rows.append({
            'harmonic': harmonic,
            'best_val_loss': best_val_loss,
            'best_val_accuracy': best_val_accuracy,
            'best_val_TopK3': best_val_top3,
            'best_val_TopK5': best_val_top5,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(by='harmonic')
    print("Summary of best validation metrics by harmonic:")
    print(summary_df)

    fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    bar_width = 40
    ax1.bar(summary_df['harmonic'], summary_df['best_val_loss'], width=bar_width, align='center', color=sns.color_palette("muted")[0])
    ax1.set_xlabel("Harmonic (Hz)", fontsize=16)
    ax1.set_ylabel("Loss", fontsize=16)
    ax1.set_title("Best Validation Loss vs. Harmonic", fontsize=18)
    ax1.set_xticks(summary_df['harmonic'])
    ax1.grid(axis='y')
    plt.tight_layout()
    fig1.savefig("best_val_loss.png", format="png", dpi=300)
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=300)
    ax2.bar(summary_df['harmonic'], summary_df['best_val_accuracy'], width=bar_width, align='center', color=sns.color_palette("muted")[1])
    ax2.set_xlabel("Harmonic (Hz)", fontsize=16)
    ax2.set_ylabel("Accuracy", fontsize=16)
    ax2.set_title("Best Validation Accuracy vs. Harmonic", fontsize=18)
    ax2.set_xticks(summary_df['harmonic'])
    ax2.grid(axis='y')
    plt.tight_layout()
    fig2.savefig("best_val_accuracy.png", format="png", dpi=300)
    plt.show()

    fig3, ax3 = plt.subplots(figsize=(10, 6), dpi=300)
    ax3.bar(summary_df['harmonic'], summary_df['best_val_TopK3'], width=bar_width, align='center', color=sns.color_palette("muted")[2])
    ax3.set_xlabel("Harmonic (Hz)", fontsize=16)
    ax3.set_ylabel("Top-3 Accuracy", fontsize=16)
    ax3.set_title("Best Validation Top-3 Accuracy vs. Harmonic", fontsize=18)
    ax3.set_xticks(summary_df['harmonic'])
    ax3.grid(axis='y')
    plt.tight_layout()
    fig3.savefig("best_val_TopK3.png", format="png", dpi=300)
    plt.show()

    fig4, ax4 = plt.subplots(figsize=(10, 6), dpi=300)
    ax4.bar(summary_df['harmonic'], summary_df['best_val_TopK5'], width=bar_width, align='center', color=sns.color_palette("muted")[3])
    ax4.set_xlabel("Harmonic (Hz)", fontsize=16)
    ax4.set_ylabel("Top-5 Accuracy", fontsize=16)
    ax4.set_title("Best Validation Top-5 Accuracy vs. Harmonic", fontsize=18)
    ax4.set_xticks(summary_df['harmonic'])
    ax4.grid(axis='y')
    plt.tight_layout()
    fig4.savefig("best_val_TopK5.png", format="png", dpi=300)
    plt.show()


if __name__ == '__main__':
    # evaluate_harmonics_for_deep_learning()
    plot_training_logs()
