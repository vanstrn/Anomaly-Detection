"""
File sets up custom callbacks useful for logging training.
"""
import tensorflow as tf


def TensorboardCallback(logDir,dataset,**kwargs):
    """Intermediate function to interface with the standard TensorBoard Callback."""
    return tf.keras.callbacks.TensorBoard(log_dir=logDir)


class CustomCallbackTemplate(tf.keras.callbacks.Callback):
    def __init__(self,logDir,dataset):
        super(CustomCallback, self).__init__()
        self.dataset=logDir
        self.dataset=dataset

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass
