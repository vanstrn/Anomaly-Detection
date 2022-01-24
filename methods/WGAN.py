"""Implementation of Wasserstein GAN, first proposed in
https://arxiv.org/abs/1701.07875

https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py
https://github.com/Mohammad-Rahmdel/WassersteinGAN-Tensorflow
provides an alternative implementation, and a more in depth summary of the method.
 """
import numpy as np
import tensorflow as tf
import logging
import time

from methods.utils import GetOptimizer
from networks.networkKeras import CreateModel
from .BaseMethod import BaseMethod

log = logging.getLogger(__name__)



class WGAN(BaseMethod):
    def __init__(self,settingsDict,dataset,networkConfig={}):
        """Initializing Model and all Hyperparameters """

        self.HPs.update({
                    "LearningRate":0.00005,
                    "LatentSize":64,
                    "Optimizer":"RMS",
                    "Epochs":10,
                    "DiscrimClipValue":0.01,
                    "GenUpdateFreq":5,
                     })

        self.requiredParams.Append([
                "GenNetworkConfig",
                "DiscNetworkConfig",
                ])

        super().__init__(settingsDict)

        #Processing Other inputs
        self.inputSpec=dataset.inputSpec
        networkConfig.update(dataset.outputSpec)
        self.Generator = CreateModel(self.HPs["GenNetworkConfig"],{"latent":self.HPs["LatentSize"]},variables=networkConfig,printSummary=True)
        self.Discriminator = CreateModel(self.HPs["DiscNetworkConfig"],dataset.inputSpec,variables=networkConfig,printSummary=True)

        self.generatorOptimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        self.discriminatorOptimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])

        self.counter=0

    def Train(self,data,callbacks=[]):
        self.InitializeCallbacks(callbacks)
        trainDataset = self.SetupDataset(data)
        for epoch in range(self.HPs["Epochs"]):
            ts = time.time()

            for batch in trainDataset:
                trainGen = (self.counter % self.HPs["GenUpdateFreq"] == 0)
                info = self.TrainStep(batch,trainGen)
                self.ClipDiscriminator()
                self.counter+=1
            self.ExecuteEpochEndCallbacks(epoch,info)
            log.info("End Epoch {}: Time {}".format(epoch,time.time()-ts))
        self.ExecuteTrainEndCallbacks({})

    @tf.function
    def TrainStep(self,images,trainGen=True):
        randomLatent = tf.random.normal([self.HPs["BatchSize"], self.HPs["LatentSize"]])

        with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
            generatedImages = self.Generator(randomLatent, training=True)["Decoder"]

            realPred = self.Discriminator(images, training=True)["Discrim"]
            fakePred = self.Discriminator({"image":generatedImages}, training=True)["Discrim"]

            genLoss = -tf.reduce_sum(fakePred)
            discLoss =  -tf.reduce_mean(realPred) + tf.reduce_mean(fakePred)

        if trainGen:
            generatorGradients = genTape.gradient(genLoss, self.Generator.trainable_variables)
            self.generatorOptimizer.apply_gradients(zip(generatorGradients, self.Generator.trainable_variables))

        discriminatorGradients = discTape.gradient(discLoss, self.Discriminator.trainable_variables)
        self.discriminatorOptimizer.apply_gradients(zip(discriminatorGradients, self.Discriminator.trainable_variables))

        return {"Generator Loss": genLoss,"Discriminator Loss": discLoss}

    @tf.function()
    def ClipDiscriminator(self):
        """
        Clips parameters of the discriminator network.

        Parameters
        ----------
        N/A


        Returns
        -------
        N/A
        """
        for params in self.Discriminator.variables:
            params.assign(tf.clip_by_value(params,-self.HPs["DiscrimClipValue"],self.HPs["DiscrimClipValue"]))

    def ImagesFromLatent(self,sample):
        return self.Generator.predict(sample)["Decoder"]
