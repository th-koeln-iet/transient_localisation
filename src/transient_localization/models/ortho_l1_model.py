import tensorflow as tf
from keras.callbacks import CSVLogger
from tensorflow.keras import layers, models, regularizers
from src.transient_localization.models.custom_regularizer import OrthogonalL1Regularizer


class SimpleAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True,
            name="attn_weight"
        )
        self.b = self.add_weight(
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
            name="attn_bias"
        )
        super(SimpleAttention, self).build(input_shape)

    def call(self, inputs):
        logits = tf.keras.backend.dot(inputs, self.W) + self.b
        attn_weights = tf.nn.softmax(tf.squeeze(logits, axis=-1), axis=1)
        attn_weights = tf.expand_dims(attn_weights, axis=-1)
        context_vector = tf.reduce_sum(inputs * attn_weights, axis=1)
        return context_vector


def build_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    inputs = layers.LayerNormalization()(inputs)
    inputs = layers.GaussianNoise(0.1)(inputs)

    x = layers.Dense(6,
                     activation='relu',
                     kernel_regularizer=OrthogonalL1Regularizer(l1=0.0001,
                                                                ortho=0.0001))(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Lambda(lambda t: tf.expand_dims(t, axis=1))(x)
    attention = SimpleAttention()(x)

    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.L2(0.0001))(attention)
    for _ in range(3):
        shortcut = x
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.L2(0.0001))(x)
        x = layers.concatenate([x, shortcut])

    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.L2(0.0001))(x)
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

    name = "full_training_150Hz_250Hz"
    x_t, y_train = load_dataset("./train_data_350Hz/")
    print(x_t.shape)
    num_samples = x_t.shape[0]
    x_train = x_t[:, [2, 4], :, :]  # Select third and fifth harmonic
    x_train = x_train.reshape((num_samples, -1))
    model = build_model(x_train.shape[1], 44 + 1)
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