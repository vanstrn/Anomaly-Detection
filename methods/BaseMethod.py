import tensorflow as tf
import logging

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
        self.logger.info("Saved model to {}".format(modelPath))

    def LoadModel(self,settings):
        """Loads specified models.  """
        modelPath = settings["modelPath"]
        try:
            self.Model = tf.keras.models.load_model(modelPath)
            self.logger.info("Loaded Keras model from {} ".format(modelPath))
        except OSError:
            self.logger.info("Could not find Keras model from {} ".format(modelPath))
        except:
            self.logger.warning("Failed to Load Keras Model from {}".format(modelPath))
