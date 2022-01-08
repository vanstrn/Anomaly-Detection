import tensorflow as tf
import logging

log = logging.getLogger(__name__)

class BaseMethod():

    def __init__():
        pass

    def Train():
        raise NotImplementedError

    def Test():
        raise NotImplementedError

    def SaveModel(self,modelPath):
        """Saves model in specified folder. Assumes """
        self.Model.save(modelPath)
        # self.Model.save(modelPath+"/saved_model.h5")
        log.info("Saved model to {}".format(modelPath))

    def LoadModel(self,settings):
        """Loads specified models.  """
        modelPath = settings["modelPath"]
        try:
            self.Model = tf.keras.models.load_model(modelPath)
            log.info("Loaded Keras model from {} ".format(modelPath))
        except OSError:
            log.info("Could not find Keras model from {} ".format(modelPath))
        except:
            log.warning("Failed to Load Keras Model from {}".format(modelPath))

    def ExecuteEpochEndCallbacks(self,epoch,logs=None):
        self.callbacks.on_epoch_end(epoch,logs)

    def ExecuteTrainEndCallbacks(self,logs=None):
        self.callbacks.on_train_end(logs)

    def InitializeCallbacks(self,callbacks):
        """Method initializes callbacks for training loops that are not `model.fit()`.
        Pass any params that the callbacks need into the generation of the callback list.

        For methods with multiple networks, pass them is as dictionaries.
        This requires callbacks that are compatible with the dictionary style of model usage.
        This style is compatible with the `method.fit()` method b/c code nests the inputed model variable without performing checks.
        Future it might be desirable to create a custom model nesting logic that will allow callbacks like `ModelCheckpoint` to be compatible.
        """
        self.callbacks = tf.keras.callbacks.CallbackList(callbacks,model=self)

    def load_weights(self):
        pass

    def save_weights(self):
        pass

    def set_weights(self):
        pass
