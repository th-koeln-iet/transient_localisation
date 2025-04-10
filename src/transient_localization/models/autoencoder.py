import tensorflow as tf
from tensorflow.keras import layers, models
from src.transient_localization.models.custom_regularizer import OrthogonalL1Regularizer


def build_autoencoder(input_dim, encoding_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim,
                           activation='relu',
                           kernel_regularizer=OrthogonalL1Regularizer(l1=0.0001, ortho=0.0001)
                          )(inputs)
    encoded = layers.BatchNormalization()(encoded)
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = models.Model(inputs, decoded)
    encoder = models.Model(inputs, encoded)
    return autoencoder, encoder


def train_autoencoder(autoencoder, x_train, epochs=50, batch_size=64):
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
