"""
File sets up custom callbacks useful for logging training.
"""
import tensorflow as tf

class LogTraining(tf.keras.callbacks.Callback):
    def __init__(self,logger,dataset):
        super(LogTraining, self).__init__()
        self.logger=logger
        self.dataset=dataset

    def on_epoch_end(self, epoch, logs=None):
        for k,v in logs.items():
            self.logger.LogScalar(tag=k,value=v,step=epoch)
