""" Implementation of ALAD from "Adversarially Learned Anomaly Detection" (ICDM 2018)
https://arxiv.org/pdf/1812.02288.pdf

Reference code: https://github.com/houssamzenati/Adversarially-Learned-Anomaly-Detection

Method Description: TBD
"""
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


class ALAD(BaseMethod):

    def __init__(self,settingsDict,dataset,networkConfig={}):
        """Initializing Model and all Hyperparameters """

        self.HPs.update({
                    "LearningRate":0.00005,
                    "LatentSize":64,
                    "Optimizer":"Adam",
                    "Epochs":10,
                    "BatchSize":32,
                    "DropRemainder":True,
                    "BetaXZ":1.0,
                    "BetaXX":1.0,
                    "BetaZZ":1.0,
                     })

        self.requiredParams.Append([ "GenNetworkConfig",
                                "DiscXZNetworkConfig",
                                "DiscXXNetworkConfig",
                                "DiscZZNetworkConfig",
                                "EncNetworkConfig",
                                ])

        #Chacking Valid hyperparameters are specified
        super().__init__(settingsDict)

        #Processing Other inputs
        self.inputSpec=dataset.inputSpec
        self.opt = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        networkConfig.update(dataset.outputSpec)
        networkConfig.update({"LatentSize":self.HPs["LatentSize"]})
        self.Generator = CreateModel(self.HPs["GenNetworkConfig"],{"latent":self.HPs["LatentSize"]},variables=networkConfig)
        self.Encoder = CreateModel(self.HPs["EncNetworkConfig"],dataset.inputSpec,variables=networkConfig)

        #Creating the mess of discriminators.
        _datasetSpec = {"z":[self.HPs["LatentSize"]],"x":dataset.inputSpec["image"]}
        self.Dxz = CreateModel(self.HPs["DiscXZNetworkConfig"],_datasetSpec,variables=networkConfig)
        _datasetSpec = {"x":dataset.inputSpec["image"],"xRec":dataset.inputSpec["image"]}
        self.Dxx = CreateModel(self.HPs["DiscXXNetworkConfig"],_datasetSpec,variables=networkConfig)
        _datasetSpec = {"z":[self.HPs["LatentSize"]],"zRec":[self.HPs["LatentSize"]]}
        self.Dzz = CreateModel(self.HPs["DiscZZNetworkConfig"],_datasetSpec,variables=networkConfig)

        # self.LoadModel({"modelPath":"models/TestVAE"})
        self.Generator.summary(print_fn=log.info)
        self.Dxz.summary(print_fn=log.info)
        self.Dxx.summary(print_fn=log.info)
        self.Dzz.summary(print_fn=log.info)
        self.Encoder.summary(print_fn=log.info)
        self.discrimVariables = self.Dxz.trainable_variables + self.Dxx.trainable_variables + self.Dzz.trainable_variables

        self.generatorOptimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        self.discriminatorOptimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        self.encoderOptimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def TrainStep(self,images):
        randomLatent = tf.random.normal([self.HPs["BatchSize"], self.HPs["LatentSize"]])
        images = images["image"]
        with tf.GradientTape() as genTape, tf.GradientTape() as discTape, tf.GradientTape() as encTape:
            fakeImage = self.Generator(randomLatent, training=True)["Decoder"]
            zReal = self.Encoder(images)["Latent"]
            fakeImage = self.Generator(zReal, training=True)["Decoder"]
            zFake = self.Encoder(images)["Latent"]

            discPredRealXZ = self.Dxz({"x":images,"z":zReal}, training=True)["Discrim"]
            discPredRealXX = self.Dxx({"x":images,"xRec":images}, training=True)["Discrim"]
            discPredRealZZ = self.Dzz({"z":randomLatent,"zRec":randomLatent}, training=True)["Discrim"]
            discPredFakeXZ = self.Dxz({"x":fakeImage,"z":randomLatent}, training=True)["Discrim"]
            discPredFakeXX = self.Dxx({"x":images,"xRec":fakeImage}, training=True)["Discrim"]
            discPredFakeZZ = self.Dzz({"z":randomLatent,"zRec":zFake}, training=True)["Discrim"]

            discLoss =  self.HPs["BetaXZ"]*self.cross_entropy(tf.ones_like(discPredRealXZ), discPredRealXZ) + \
                        self.HPs["BetaXX"]*self.cross_entropy(tf.ones_like(discPredRealXX), discPredRealXX) + \
                        self.HPs["BetaZZ"]*self.cross_entropy(tf.ones_like(discPredRealZZ), discPredRealZZ) + \
                        self.HPs["BetaXZ"]*self.cross_entropy(tf.zeros_like(discPredFakeXZ), discPredFakeXZ) + \
                        self.HPs["BetaXX"]*self.cross_entropy(tf.zeros_like(discPredFakeXX), discPredFakeXX) + \
                        self.HPs["BetaZZ"]*self.cross_entropy(tf.zeros_like(discPredFakeZZ), discPredFakeZZ)
            _genLoss = self.cross_entropy(tf.zeros_like(discPredFakeXZ), discPredFakeXZ)
            _encLoss = self.cross_entropy(tf.ones_like(discPredRealXZ), discPredRealXZ)

            cycleLoss = self.HPs["BetaXX"]*self.cross_entropy(tf.ones_like(discPredRealXX), discPredRealXX) + \
                        self.HPs["BetaZZ"]*self.cross_entropy(tf.ones_like(discPredRealZZ), discPredRealZZ) + \
                        self.HPs["BetaXX"]*self.cross_entropy(tf.zeros_like(discPredFakeXX), discPredFakeXX) + \
                        self.HPs["BetaZZ"]*self.cross_entropy(tf.zeros_like(discPredFakeZZ), discPredFakeZZ)
            genLoss = _genLoss + cycleLoss
            encLoss = _encLoss + cycleLoss

        generatorGradients = genTape.gradient(genLoss, self.Generator.trainable_variables)
        encoderGradients = encTape.gradient(encLoss, self.Encoder.trainable_variables)
        discriminatorGradients = discTape.gradient(discLoss, self.discrimVariables)

        self.generatorOptimizer.apply_gradients(zip(generatorGradients, self.Generator.trainable_variables))
        self.encoderOptimizer.apply_gradients(zip(encoderGradients, self.Encoder.trainable_variables))
        self.discriminatorOptimizer.apply_gradients(zip(discriminatorGradients, self.discrimVariables))

        return {"Generator Loss": genLoss,"Discriminator Loss": discLoss,"Encoder Loss": encLoss, "Cycle Loss": cycleLoss}

    def Test(self,data):
        pass

    def InitializeCallbacks(self,callbacks):
        """Method initializes callbacks for training loops that are not `model.fit()`.
        Additional logic is still required for methods with multiple networks, such as GANs."""
        self.callbacks = tf.keras.callbacks.CallbackList(callbacks,model=self)

    def ImagesFromLatent(self,sample):
        return self.Generator.predict(sample)["Decoder"]

    def LatentFromImage(self,sample):
        return self.Encoder.predict(sample)["Latent"]

    def ImagesFromImage(self,testImages):
        z = self.Encoder.predict({"image":testImages})["Latent"]
        return self.Generator.predict({"latent":z})["Decoder"]

    def ImageDiscrim(self,testImages):
        z = self.Encoder.predict({"image":testImages})["Latent"]
        return self.Dxz.predict({"x":testImages,"z":z})["Discrim"]

    def AnomalyScore(self,testImages,alpha=0.9):
        v1 = tf.reduce_sum((testImages-self.ImagesFromImage(testImages))**2,axis=list(range(1,len(testImages.shape))))
        v2 = tf.squeeze(self.ImageDiscrim(testImages))
        return alpha * v1 + (1-alpha)*v2
