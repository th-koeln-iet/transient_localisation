from src.transient_localization.data_handling.dataset import load_dataset
from src.transient_localization.models.autoencoder import build_autoencoder, train_autoencoder

x_train, y_train = load_dataset()

num_samples = x_train.shape[0]
x_train_flat = x_train[:, [1, 3, 5], :, :].reshape(num_samples, -1)

autoencoder, encoder = build_autoencoder(input_dim=x_train_flat.shape[1], encoding_dim=5)
train_autoencoder(autoencoder, x_train_flat, epochs=50, batch_size=64)
