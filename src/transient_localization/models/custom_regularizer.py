import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import Regularizer


class OrthogonalL1Regularizer(Regularizer):
    def __init__(self, l1=0.001, ortho=0.001):
        self.l1 = l1
        self.ortho = ortho

    def __call__(self, x):
        l1_loss = self.l1 * K.sum(K.abs(x))
        x_shape = K.int_shape(x)
        if len(x_shape) < 2:
            return l1_loss
        identity = tf.eye(x_shape[-1], dtype=x.dtype)
        wt_w = K.dot(K.transpose(x), x)
        ortho_loss = self.ortho * K.sum(K.square(wt_w - identity))
        return l1_loss + ortho_loss

    def get_config(self):
        return {'l1': self.l1, 'ortho': self.ortho}
