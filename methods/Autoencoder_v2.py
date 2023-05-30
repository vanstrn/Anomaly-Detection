#!/usr/bin/env python
# Created By  : Neale Van Stralen
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Method definitions for alternative network and training methods for autoencoder-based methods.
Primarily used for performance testing of different experimental setups.
"""

import tensorflow as tf
import logging

from methods.utils import GetOptimizer
from networks.networkKeras import CreateModel
from .BaseMethod import BaseMethod

log = logging.getLogger(__name__)

class Autoencoder_v2(BaseMethod):
    def __init__(self,settingsDict,dataset,networkConfig={}):
        """Autoencoder test method. This implementation runs about 5-10% slower than the base Autoencoding method."""

        self.hyperParams.update({
                    "LearningRate":0.00005,
                    "Optimizer":"Adam",
                    "Epochs":10,
                    "LatentSize":64,
                    "BatchSize":64,
                    "Shuffle":True,
                     })

        self.requiredParams.Append(["NetworkConfig",
                          ])

        super().__init__(settingsDict)

        #Processing Other inputs
        self.opt = GetOptimizer(self.hyperParams["Optimizer"],self.hyperParams["LearningRate"])
        networkConfig.update(dataset.outputSpec)
        networkConfig.update({"LatentSize":self.hyperParams["LatentSize"]})
        self.Model = CreateModel(self.hyperParams["NetworkConfig"],dataset.inputSpec,variables=networkConfig,printSummary=True)
        self.Model.compile(optimizer=self.opt, loss=["mse"],metrics=[])

    def Train(self,data,callbacks=[]):
        self.InitializeCallbacks(callbacks)
        self.Model.fit( data["image"],
                        data["image"],
                        epochs=self.hyperParams["Epochs"],
                        batch_size=self.hyperParams["BatchSize"],
                        shuffle=self.hyperParams["Shuffle"],
                        callbacks=self.callbacks)
        self.SaveModel("models/TestAE")

    def ImagesFromImage(self,testImages):
        return self.Model.predict({"image":testImages},verbose=0)["Decoder"]

    def AnomalyScore(self,testImages):
        return tf.reduce_sum((testImages-self.ImagesFromImage(testImages))**2,axis=list(range(1,len(testImages.shape))))


class Autoencoder_v3(BaseMethod):
    def __init__(self,settingsDict,dataset,networkConfig={}):
        """Autoencoder test method. This implementation runs about 4-8% slower than the base Autoencoding method."""

        self.hyperParams.update({
                    "LearningRate":0.00005,
                    "LatentSize":64,
                    "Optimizer":"Adam",
                    "Epochs":10,
                    "BatchSize":64,
                     })

        self.requiredParams.Append([
                "GenNetworkConfig",
                "EncNetworkConfig",
                ])

        super().__init__(settingsDict)

        #Processing Other inputs
        self.inputSpec=dataset.inputSpec
        networkConfig.update(dataset.outputSpec)
        networkConfig.update({"LatentSize":self.hyperParams["LatentSize"]})
        self.Generator = CreateModel(self.hyperParams["GenNetworkConfig"],{"latent":self.hyperParams["LatentSize"]},variables=networkConfig,printSummary=True)
        self.Encoder = CreateModel(self.hyperParams["EncNetworkConfig"],dataset.inputSpec,variables=networkConfig,printSummary=True)

        self.optimizer = GetOptimizer(self.hyperParams["Optimizer"],self.hyperParams["LearningRate"])
        self.mse = tf.keras.losses.MeanSquaredError()

    @tf.function
    def TrainStep(self,images,hyperParams):

        with tf.GradientTape() as tape:
            latent = self.Encoder(images, training=True)["Latent"]
            generatedImages = self.Generator(latent, training=True)["Decoder"]

            loss = self.mse(images["image"],generatedImages)

        gradients = tape.gradient(loss, self.Generator.trainable_variables+self.Encoder.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.Generator.trainable_variables+self.Encoder.trainable_variables))

        return {"Autoencoder Loss": loss}

    def ImagesFromLatent(self,sample):
        return self.Generator.predict(sample,verbose=0)["Decoder"]

    def ImagesFromImage(self,testImages):
        latent=self.Encoder.predict({"image":testImages},verbose=0)["Latent"]
        return self.Generator.predict({"latent":latent},verbose=0)["Decoder"]

    def AnomalyScore(self,testImages):
        return tf.reduce_sum((testImages-self.ImagesFromImage(testImages))**2,axis=list(range(1,len(testImages.shape))))

    def LatentFromImage(self,sample):
        return self.Encoder.predict(sample,verbose=0)["Latent"]
