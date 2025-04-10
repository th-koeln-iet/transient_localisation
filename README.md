Orthogonal & Sparse Autoencoder Pretraining

This project implements a pipeline that:
1. Transforms simulation CSV files into a dataset containing multiple harmonic orders.
2. Optionally applies PCA to reduce dimensionality.
3. Pretrains a custom autoencoder with an orthogonal & L1 regularized bottleneck layer.


1. Create a virtual environment and install the requirements:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/MacOS
    venv\Scripts\activate     # Windows
    pip install -r requirements.txt
    ```

2. Run the main script:
    ```bash
    python main.py
    ```

## Functionality

- **Data Transformation:** See `src/data/transform.py` and `src/data/dataset.py`
- **PCA Pretraining:** See `src/models/pca_pretrain.py`
- **Autoencoder Pretraining:** See `src/models/autoencoder.py`
- **Custom Regularizer:** See `src/models/custom_regularizers.py`

