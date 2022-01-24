
import numpy as np
import tensorflow as tf
import logging
import matplotlib.pyplot as plt
import time

from utils.utils import CheckFilled
from methods.utils import GetOptimizer
from networks.networkKeras import CreateModel
from .BaseMethod import BaseMethod

log = logging.getLogger(__name__)

from .GAN import GAN

class AnoGAN(GAN):
    def __init__(self,settingsDict,dataset,networkConfig={}):
        """Initializing Model and all Hyperparameters """

        self.HPs.update({
            "PredBatchSize":16,
            "AnomalyFitEpochs":10,
            })
        self.requiredParams.Append(["LatentNetworkConfig"])
        super().__init__(settingsDict,dataset,networkConfig)

    def GenerateLatent(self,sample):
        #Building the latent network
        LatentPredNet = CreateModel(self.HPs["LatentNetworkConfig"],self.inputSpec,variables={"LatentSize":self.HPs["LatentSize"]})
        latentOptimizer = tf.keras.optimizers.Adam(1e-4)
        #Training the latent network
        for _ in range(self.HPs["AnomalyFitEpochs"]):
            with tf.GradientTape() as tape:
                z = LatentPredNet(sample)["Latent"]
                out = self.Generator(z)["Decoder"]
                latentLoss = tf.math.abs(out-sample)
            latentGradients = tape.gradient(latentLoss, LatentPredNet.trainable_variables)
            latentOptimizer.apply_gradients(zip(latentGradients, LatentPredNet.trainable_variables))

        return LatentPredNet(sample)["Latent"]

    def ImagesFromImage(self,sample):
        imagePred = np.zeros_like(sample)
        #Spliting data into appropriate batches and running them through the things
        i=0
        testDataset = tf.data.Dataset.from_tensor_slices(sample).batch(self.HPs["PredBatchSize"])
        for batch in testDataset:
            z = self.GenerateLatent(batch)
            tmp = self.Generator.predict(z)["Decoder"]
            imagePred[i:i+batch.shape[0],:] = tmp
            i+=batch.shape[0]
        return imagePred

    def AnomalyScore(self,testImages):
        return tf.reduce_sum((testImages-self.ImagesFromImage(testImages))**2,axis=list(range(1,len(testImages.shape))))
