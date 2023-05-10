
import tensorflow as tf
import logging

from methods.utils import GetOptimizer
from networks.networkKeras import CreateModel
from .BaseMethod import BaseMethod

log = logging.getLogger(__name__)


class Autoencoder(BaseMethod):
    def __init__(self,settingsDict,dataset,networkConfig={}):
        """Initializing Model and all Hyperparameters """


        self.HPs.update({
                    "LearningRate":0.001,
                    "Optimizer":"Adam",
                    "Epochs":10,
                    "BatchSize":64,
                     })

        self.requiredParams.Append([
                "NetworkConfig",
                ])

        super().__init__(settingsDict)

        #Processing Other inputs
        self.inputSpec=dataset.inputSpec
        networkConfig.update(dataset.outputSpec)
        self.Model = CreateModel(self.HPs["NetworkConfig"],dataset.inputSpec,variables=networkConfig,printSummary=True)

        self.optimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        self.mse = tf.keras.losses.MeanSquaredError()

    @tf.function
    def TrainStep(self,images):

        with tf.GradientTape() as tape:
            generatedImages = self.Model(images, training=True)["Decoder"]

            loss = self.mse(images["image"],generatedImages)

        gradients = tape.gradient(loss, self.Model.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.Model.trainable_variables))

        return {"Autoencoder Loss": loss}


    def ImagesFromImage(self,testImages):
        return self.Model.predict({"image":testImages})["Decoder"]

    def LatentFromImage(self,testImages):
        return self.Model.predict({"image":testImages})["Latent"]

    def AnomalyScore(self,testImages):
        return tf.reduce_sum((testImages-self.ImagesFromImage(testImages))**2,axis=list(range(1,len(testImages.shape))))
