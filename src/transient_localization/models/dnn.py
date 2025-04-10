import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    inputs = layers.LayerNormalization()(inputs)
    inputs = layers.GaussianNoise(0.1)(inputs)

    x = layers.Dense(64, activation='relu')(inputs)
    for _ in range(3):
        shortcut = x
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.concatenate([x, shortcut])

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model
