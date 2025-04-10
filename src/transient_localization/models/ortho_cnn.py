import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from keras.callbacks import CSVLogger


def build_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    inputs = layers.LayerNormalization()(inputs)
    inputs = layers.GaussianNoise(0.1)(inputs)

    x = layers.Conv2D(3, kernel_size=(1, 1),
                      padding='same',
                      activation='relu',
                      kernel_regularizer=regularizers.L1(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(6, kernel_size=(2, 2), padding='same', activation='relu', kernel_regularizer=regularizers.L2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(9, kernel_size=(2, 2), padding='same', activation='relu', kernel_regularizer=regularizers.L2(0.001))(x)
    x = layers.BatchNormalization()(x)

    attention = layers.MultiHeadAttention(num_heads=2, key_dim=2, kernel_regularizer=regularizers.L2(0.001))(x, x)
    attention = layers.Flatten()(attention)

    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.L2(0.0001))(attention)
    for _ in range(3):
        shortcut = x
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.L2(0.0001))(x)
        x = layers.concatenate([x, shortcut])

    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.L2(0.0001))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.L2(0.0001))(x)

    model = models.Model(inputs, outputs)
    return model


if __name__ == '__main__':
    from src.transient_localization.data_handling.dataset import load_dataset
    import keras.callbacks
    from keras import metrics
    from src.transient_localization.utils.sgdr_scheduler import SGDRScheduler
    from tensorflow.keras import optimizers
    import numpy as np

    name = "full_training_cnn_150Hz_250Hz"
    x_t, y_train = load_dataset("./train_data_350Hz/")
    print(x_t.shape)
    num_samples = x_t.shape[0]
    x_train = x_t[:, [2, 4], :, :]
    model = build_model((2, 3, 2), 44 + 1)
    batch_size = np.ceil(float(x_t.shape[0]) / 1)
    top_k3 = metrics.SparseTopKCategoricalAccuracy(k=3, name="TopK3")
    top_k5 = metrics.SparseTopKCategoricalAccuracy(k=5, name="TopK5")

    chp = keras.callbacks.ModelCheckpoint(filepath=f"{name}.h5",
                                          monitor='val_accuracy',
                                          verbose=1,
                                          save_weights_only=True,
                                          save_best_only=True,
                                          initial_value_threshold=0.0)
    csv_logger = CSVLogger(f'./logs/{name}.log')
    schedule = SGDRScheduler(min_lr=1e-8,
                             max_lr=1e-4,
                             steps_per_epoch=np.ceil(len(x_train * 0.9) / batch_size),
                             lr_decay=0.8,
                             cycle_length=1000,
                             mult_factor=1.5)

    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', top_k3, top_k5])

    model.summary()

    model.load_weights(f"{name}.h5")
    hist = model.fit(x_train, y_train,
                     batch_size=int(batch_size),
                     validation_split=0.1,
                     epochs=1000000,
                     callbacks=[chp, schedule])