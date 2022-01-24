""" Implementation of BiGAN from "Adversarial Feature Learning" (ICLR 2017)
https://arxiv.org/abs/1605.09782

This method is also the same as "Adversarially Learned Inference" (ICLR 2017):
https://arxiv.org/abs/1606.00704

When used for anomaly detection it becomes "Efficient GAN-Based Anomaly Detection" (arxiv 2018 / Workshop ICLR 2018):
https://arxiv.org/abs/1802.06222

Method Description: TBD
"""
import numpy as np
import tensorflow as tf
import logging

from methods.utils import GetOptimizer
from networks.networkKeras import CreateModel
from .BaseMethod import BaseMethod

log = logging.getLogger(__name__)


class BiGAN(BaseMethod):
    def __init__(self,settingsDict,dataset,networkConfig={}):
        """Initializing Model and all Hyperparameters """

        self.HPs.update({
                    "LearningRate":0.00005,
                    "LatentSize":64,
                    "Optimizer":"Adam",
                    "Epochs":10,
                    "BatchSize":32,
                    "Shuffle":True,
                     })

        self.requiredParams.Append([
            "GenNetworkConfig",
            "DiscNetworkConfig",
            "EncNetworkConfig",
            ])

        super().__init__(settingsDict)

        #Processing Other inputs
        self.inputSpec=dataset.inputSpec
        self.opt = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        networkConfig.update(dataset.outputSpec)
        networkConfig.update({"LatentSize":self.HPs["LatentSize"]})
        self.Generator = CreateModel(self.HPs["GenNetworkConfig"],{"latent":self.HPs["LatentSize"]},variables=networkConfig,printSummary=True)
        self.Encoder = CreateModel(self.HPs["EncNetworkConfig"],dataset.inputSpec,variables=networkConfig,printSummary=True)
        _datasetSpec = {"features":[self.HPs["LatentSize"]],**dataset.inputSpec}
        self.Discriminator = CreateModel(self.HPs["DiscNetworkConfig"],_datasetSpec,variables=networkConfig,printSummary=True)

        self.generatorOptimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        self.discriminatorOptimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        self.crossEntropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def TrainStep(self,images):
        randomLatent = tf.random.normal([self.HPs["BatchSize"], self.HPs["LatentSize"]])

        with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
            generatedImages = self.Generator(randomLatent, training=True)["Decoder"]

            e_z = self.Encoder(images)["Latent"]
            realPred = self.Discriminator({**images,"features":e_z}, training=True)["Discrim"]
            fakePred = self.Discriminator({"image":generatedImages,"features":randomLatent}, training=True)["Discrim"]

            discLoss = self.crossEntropy(tf.ones_like(realPred), realPred) + \
                        self.crossEntropy(tf.zeros_like(fakePred), fakePred)
            genLoss = self.crossEntropy(tf.ones_like(fakePred), fakePred)
            encLoss = self.crossEntropy(tf.zeros_like(realPred), realPred)
            genAllLoss = genLoss + encLoss

        generatorGradients = genTape.gradient(genAllLoss, self.Generator.trainable_variables+self.Encoder.trainable_variables)
        discriminatorGradients = discTape.gradient(discLoss, self.Discriminator.trainable_variables)

        self.generatorOptimizer.apply_gradients(zip(generatorGradients, self.Generator.trainable_variables+self.Encoder.trainable_variables))
        self.discriminatorOptimizer.apply_gradients(zip(discriminatorGradients, self.Discriminator.trainable_variables))

        return {"Generator Loss": gen_loss,"Discriminator Loss": disc_loss,"Encoder Loss": enc_loss}

    def ImagesFromLatent(self,sample):
        return self.Generator.predict(sample)["Decoder"]

    def ImagesFromImage(self,testImages):
        z = self.Encoder.predict({"image":testImages})["Latent"]
        return self.Generator.predict({"latent":z})["Decoder"]

    def ImageDiscrim(self,testImages):
        z = self.Encoder.predict({"image":testImages})["Latent"]
        return self.Discriminator.predict({"image":testImages,"features":z})["Discrim"]

    def AnomalyScore(self,testImages,alpha=0.9):
        v1 = tf.reduce_sum((testImages-tf.squeeze(self.ImagesFromImage(testImages)))**2,axis=[1,2])
        v2 = tf.squeeze(self.ImageDiscrim(testImages))
        return alpha * v1 + (1-alpha)*v2
