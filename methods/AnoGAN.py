
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

        super().__init__(settingsDict,dataset,networkConfig)
        self.HPs.update({"AnomalyFitEpochs":10,"LatentNetworkConfig":"netConfigs/GAN/Encoder.json"})

    def GenerateLatent(self,sample):
        #Building the latent network
        LatentPredNet = CreateModel(self.HPs["LatentNetworkConfig"],self.inputSpec,variables={"LatentSize":self.HPs["LatentSize"]})
        #Setting up the loss functions.
        latent_optimizer = tf.keras.optimizers.Adam(1e-4)
        #Training the latent network
        for _ in range(self.HPs["AnomalyFitEpochs"]):
            with tf.GradientTape() as tape:
                z = LatentPredNet(sample)["Latent"]
                out = tf.squeeze(self.Generator(z)["Decoder"])
                # print(out.shape)
                # print(sample.shape)
                latent_loss = tf.math.abs(out-sample)
            gradients_of_latent = tape.gradient(latent_loss, LatentPredNet.trainable_variables)
            latent_optimizer.apply_gradients(zip(gradients_of_latent, LatentPredNet.trainable_variables))

        #Returning the final z
        return LatentPredNet(sample)["Latent"]

    def ImagesFromImage(self,sample):
        imagePred = np.zeros_like(sample)
        #Spliting data into appropriate batches and running them through the things
        i=0
        testDataset = tf.data.Dataset.from_tensor_slices(sample).batch(self.HPs["BatchSize"])
        for batch in testDataset:
            z = self.GenerateLatent(batch)
            tmp = tf.squeeze(self.Generator.predict(z)["Decoder"])
            imagePred[i:i+batch.shape[0],:] = tmp
            i+=batch.shape[0]
        return imagePred

    def AnomalyScore(self,testImages):
        return tf.reduce_sum((testImages-tf.squeeze(self.ImagesFromImage(testImages)))**2,axis=[1,2])
