from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import math


class SGDRScheduler(Callback):
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay
        self.lr = max_lr

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        self.lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * fraction_to_restart))
        return self.lr

    def on_train_begin(self, logs={}):
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_begin(self, epoch, logs={}):
        print(f"Current Learning Rate: {self.lr}")

    def on_epoch_end(self, epoch, logs={}):
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = math.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        self.model.set_weights(self.best_weights)