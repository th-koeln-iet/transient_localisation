from src.transient_localization.data_handling.dataset import load_dataset
from keras import metrics
from src.transient_localization.utils.sgdr_scheduler import SGDRScheduler
from tensorflow.keras import optimizers
from src.transient_localization.models.dnn import build_model
from keras.callbacks import CSVLogger
import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

    bar_width = 40

    plt.figure()
    plt.bar(summary_df['harmonic'], summary_df['best_val_loss'], width=bar_width, align='center')
    plt.xlabel("Harmonic (Hz)")
    plt.ylabel("Loss")
    plt.title("Best Validation Loss vs. Harmonic")
    plt.xticks(summary_df['harmonic'])
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.bar(summary_df['harmonic'], summary_df['best_val_accuracy'], width=bar_width, align='center')
    plt.xlabel("Harmonic (Hz)")
    plt.ylabel("Accuracy")
    plt.title("Best Validation Accuracy vs. Harmonic")
    plt.xticks(summary_df['harmonic'])
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.bar(summary_df['harmonic'], summary_df['best_val_TopK3'], width=bar_width, align='center')
    plt.xlabel("Harmonic (Hz)")
    plt.ylabel("Top-3 Accuracy")
    plt.title("Best Validation Top-3 Accuracy vs. Harmonic")
    plt.xticks(summary_df['harmonic'])
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.bar(summary_df['harmonic'], summary_df['best_val_TopK5'], width=bar_width, align='center')
    plt.xlabel("Harmonic (Hz)")
    plt.ylabel("Top-5 Accuracy")
    plt.title("Best Validation Top-5 Accuracy vs. Harmonic")
    plt.xticks(summary_df['harmonic'])
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    evaluate_harmonics_for_deep_learning()
    plot_training_logs()

