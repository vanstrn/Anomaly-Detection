
import numpy as np
import tensorflow as tf
import logging
import time

from methods.utils import GetOptimizer
from networks.networkKeras import CreateModel
from .BaseMethod import BaseMethod

log = logging.getLogger(__name__)



class GANomaly(BaseMethod):
    def __init__(self,settingsDict,dataset,networkConfig={}):
        """Initializing Model and all Hyperparameters """

        self.HPs.update({
                    "LearningRate":0.00005,
                    "LatentSize":16,
                    "Optimizer":"Adam",
                    "Epochs":10,
                    "w_adv":1,
                    "w_con":50,
                    "w_enc":1,
                     })

        self.requiredParams.Append([
                "GenNetworkConfig",
                "EncNetworkConfig",
                "DiscNetworkConfig",
                                ])

        super().__init__(settingsDict)

        #Processing Other inputs
        networkConfig.update(dataset.outputSpec)
        networkConfig.update({"LatentSize":self.HPs["LatentSize"]})
        self.Generator = CreateModel(self.HPs["GenNetworkConfig"],{"latent":self.HPs["LatentSize"]},variables=networkConfig,printSummary=True)
        self.Discriminator = CreateModel(self.HPs["DiscNetworkConfig"],dataset.inputSpec,variables=networkConfig,printSummary=True)
        self.Encoder = CreateModel(self.HPs["EncNetworkConfig"],dataset.inputSpec,variables=networkConfig,printSummary=True)
        self.Encoder2 = CreateModel(self.HPs["EncNetworkConfig"],dataset.inputSpec,variables=networkConfig)

        self.generatorOptimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        self.discriminatorOptimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        self.encoderOptimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        self.crossEntropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def Train(self,data,callbacks=[]):
        self.InitializeCallbacks(callbacks)
        trainDataset = self.SetupDataset(data)
        for epoch in range(self.HPs["Epochs"]):
            ts = time.time()

            for batch in trainDataset:
                info1={}
                info2 = self.TrainStepXZX(batch)
            self.ExecuteEpochEndCallbacks(epoch,{**info1,**info2})
            log.info("End Epoch {}: Time {}".format(epoch,time.time()-ts))
        self.ExecuteTrainEndCallbacks({})

    @tf.function
    def TrainStepGAN(self,images):
        randomLatent = tf.random.normal([self.HPs["BatchSize"], self.HPs["LatentSize"]])

        with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
            generatedImages = self.Generator(randomLatent, training=True)["Decoder"]

            realOutput = self.Discriminator(images, training=True)
            fakeOutput = self.Discriminator({"image":generatedImages}, training=True)
            realPred = realOutput["Discrim"]; realFeatures = realOutput["Features"]
            fakePred = fakeOutput["Discrim"]; fakeFeatures = fakeOutput["Features"]

            genLoss = self.crossEntropy(tf.ones_like(fakePred), fakePred)
            featLoss = tf.reduce_mean(realFeatures-fakeFeatures)**2.0
            discLoss = self.crossEntropy(tf.ones_like(realPred), realPred) + \
                        self.crossEntropy(tf.zeros_like(fakePred), fakePred)
            genAllLoss = genLoss # + feat_loss

        generatorGradients = genTape.gradient(genAllLoss, self.Generator.trainable_variables)
        discriminatorGradients = discTape.gradient(discLoss, self.Discriminator.trainable_variables)

        self.generatorOptimizer.apply_gradients(zip(generatorGradients, self.Generator.trainable_variables))
        self.discriminatorOptimizer.apply_gradients(zip(discriminatorGradients, self.Discriminator.trainable_variables))

        return {"Generator Loss": genLoss,"Discriminator Loss": discLoss}

    # @tf.function
    def TrainStepXZX(self,images):

        with tf.GradientTape() as genTape, tf.GradientTape() as encTape, tf.GradientTape() as discTape:
            z = self.Encoder(images, training=True)["Latent"]
            x_hat = self.Generator(z, training=True)["Decoder"]
            z_hat = self.Encoder2(x_hat, training=True)["Latent"]

            realOutput = self.Discriminator(images, training=True)
            fakeOutput = self.Discriminator(x_hat, training=True)
            realPred = realOutput["Discrim"]; realFeatures = realOutput["Features"]
            fakePred = fakeOutput["Discrim"]; fakeFeatures = fakeOutput["Features"]

            genLoss = self.crossEntropy(tf.ones_like(fakePred), fakePred)
            featLoss = tf.reduce_mean(realFeatures-fakeFeatures)**2.0
            discLoss = self.crossEntropy(tf.ones_like(realPred), realPred) + \
                        self.crossEntropy(tf.zeros_like(fakePred), fakePred)

            encoderLoss = (z_hat-z)**2
            contextLoss = tf.math.abs(tf.squeeze(x_hat)-images["image"])
            totalLoss = tf.reduce_mean(encoderLoss) +tf.reduce_mean(contextLoss) + genLoss + featLoss

        generatorGradients = genTape.gradient(totalLoss, self.Generator.trainable_variables)
        discriminatorGradients = discTape.gradient(totalLoss, self.Discriminator.trainable_variables)
        encoderGradients = encTape.gradient(totalLoss, self.Encoder.trainable_variables+self.Encoder2.trainable_variables)

        self.generatorOptimizer.apply_gradients(zip(generatorGradients, self.Generator.trainable_variables))
        self.encoderOptimizer.apply_gradients(zip(encoderGradients, self.Encoder.trainable_variables+self.Encoder2.trainable_variables))
        self.discriminatorOptimizer.apply_gradients(zip(discriminatorGradients, self.Discriminator.trainable_variables))

        return {"Generator Loss": genLoss,"Discriminator Loss": discLoss,"Feature Loss":featLoss,"Encoding Loss": tf.reduce_mean(encoderLoss),"Construction Loss": tf.reduce_mean(contextLoss)}

    def InitializeCallbacks(self,callbacks):
        """Method initializes callbacks for training loops that are not `model.fit()`.
        Pass any params that the callbacks need into the generation of the callback list.

        For methods with multiple networks, pass them is as dictionaries.
        This requires callbacks that are compatible with the dictionary style of model usage.
        This style is compatible with the `method.fit()` method b/c code nests the inputed model variable without performing checks.
        Future it might be desirable to create a custom model nesting logic that will allow callbacks like `ModelCheckpoint` to be compatible.
        """
        self.callbacks = tf.keras.callbacks.CallbackList(callbacks,model=self,LatentSize=self.HPs["LatentSize"])

    def LatentFromImage(self,sample):
        return self.Generator.predict(sample)

    def ImagesFromLatent(self,sample):
        return self.Generator.predict(sample)["Decoder"]

    def ImagesFromImage(self,testImages):
        z = self.Encoder.predict({"image":testImages})["Latent"]
        return self.Generator.predict({"latent":z})["Decoder"]

    def AnomalyScore(self,testImages):
        return tf.reduce_sum((testImages-tf.squeeze(self.ImagesFromImage(testImages)))**2,axis=[1,2])
