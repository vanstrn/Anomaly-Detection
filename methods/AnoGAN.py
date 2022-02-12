
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
            "AnomalyFitEpochs":50,
            "OptimizerLatent":"Adam",
            "LearningRateLatent":1E-3
            })
        super().__init__(settingsDict,dataset,networkConfig)

    def GenerateLatent(self,sample):
        latentOptimizer = GetOptimizer(self.HPs["OptimizerLatent"],self.HPs["LearningRateLatent"])
        z = tf.Variable(initial_value=tf.random.normal([sample.shape[0], self.HPs["LatentSize"]]),trainable=True)
        for _ in range(self.HPs["AnomalyFitEpochs"]):
            with tf.GradientTape() as tape:
                out = self.Generator(z)["Decoder"]
                latentLoss = tf.math.abs(out-sample)
            latentGradients = tape.gradient(latentLoss, [z])
            latentOptimizer.apply_gradients(zip(latentGradients, [z]))
        return z

    def ImagesFromImage(self,sample):
        imagePredList = []
        #Spliting data into appropriate batches and running them through the things
        testDataset = tf.data.Dataset.from_tensor_slices(sample).batch(self.HPs["PredBatchSize"])
        for batch in testDataset:
            z = self.GenerateLatent(batch)
            out=self.Generator(z)["Decoder"]
            imagePredList.append(out)
        return tf.concat(imagePredList,axis=0).numpy()

    def DiscrimAnomaly(self,testImages):
        return self.Discriminator.predict({"image":testImages})["Discrim"]

    def ImageAnomaly(self,testImages):
        return tf.reduce_sum((testImages-self.ImagesFromImage(testImages))**2,axis=list(range(1,len(testImages.shape))))

    def AnomalyScore(self,testImages,alpha=0.9):
        return tf.reduce_sum((testImages-self.ImagesFromImage(testImages))**2,axis=list(range(1,len(testImages.shape))))

class AnoGAN_v2(GAN):
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

    def DiscrimAnomaly(self,testImages):
        z = self.Encoder.predict({"image":testImages})["Latent"]
        return self.Discriminator.predict({"image":testImages,"features":z})["Discrim"]

    def ImageAnomaly(self,testImages):
        return tf.reduce_sum((testImages-self.ImagesFromImage(testImages))**2,axis=list(range(1,len(testImages.shape))))

    def AnomalyScore(self,testImages,alpha=0.9):
        return tf.reduce_sum((testImages-self.ImagesFromImage(testImages))**2,axis=list(range(1,len(testImages.shape))))
