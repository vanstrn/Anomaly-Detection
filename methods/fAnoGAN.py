""" Implementation of fAnoGAN from "f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks." (Medical Image Analysis 2019)


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
from .BiGAN import BiGAN

log = logging.getLogger(__name__)


class fAnoGAN(BiGAN):
    def __init__(self,*args,**kwargs):
        """Initializing Model and all Hyperparameters """
        super().__init__(*args,**kwargs)
        self.mse = tf.keras.losses.MeanSquaredError()

    @tf.function
    def TrainStep(self,images):
        randomLatent = tf.random.normal([self.HPs["BatchSize"], self.HPs["LatentSize"]])

        with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
            generatedImages = self.Generator(randomLatent, training=True)["Decoder"]

            e_z = self.Encoder(images)["Latent"]

            out= self.Discriminator({**images,"features":e_z}, training=True)
            realPred = out["Discrim"]
            realFeatures = out["Features"]
            fakePred = self.Discriminator({"image":generatedImages,"features":randomLatent}, training=True)["Discrim"]

            discLoss = self.crossEntropy(tf.ones_like(realPred), realPred) + \
                        self.crossEntropy(tf.zeros_like(fakePred), fakePred)
            genLoss = self.crossEntropy(tf.ones_like(fakePred), fakePred)
            encLoss = self.crossEntropy(tf.zeros_like(realPred), realPred)

            # ziz Training
            e_z_fake = self.Encoder(generatedImages)["Latent"]
            zizLoss = self.mse(randomLatent, e_z_fake)

            # izi Training
            reconImage = self.Generator(e_z, training=True)["Decoder"]
            fakeFeatures = self.Discriminator({"image":reconImage,"features":e_z}, training=True)["Features"]
            iziLoss = self.mse(images["image"], tf.squeeze(reconImage))
            iziFeatureLoss = self.mse(realFeatures, fakeFeatures)

            totalloss = genLoss + encLoss + zizLoss + iziLoss + iziFeatureLoss

        generatorGradients = genTape.gradient(totalloss, self.Generator.trainable_variables+self.Encoder.trainable_variables)
        discriminatorGradients = discTape.gradient(discLoss, self.Discriminator.trainable_variables)

        self.generatorOptimizer.apply_gradients(zip(generatorGradients, self.Generator.trainable_variables+self.Encoder.trainable_variables))
        self.discriminatorOptimizer.apply_gradients(zip(discriminatorGradients, self.Discriminator.trainable_variables))

        return {"Generator Loss": genLoss,"Discriminator Loss": discLoss,"Encoder Loss": encLoss}

    def DiscrimAnomaly(self,testImages):
        z = self.Encoder.predict({"image":testImages})["Latent"]
        return self.Discriminator.predict({"image":testImages,"features":z})["Discrim"]

    def ImageAnomaly(self,testImages):
        return tf.reduce_sum((testImages-tf.squeeze(self.ImagesFromImage(testImages)))**2,axis=[1,2])

    def FeatureAnomaly(self,testImages):
        z = self.Encoder.predict({"image":testImages})["Latent"]
        featuresOrig = self.Discriminator.predict({"image":testImages,"features":z})["Features"]
        genImage = self.Generator.predict({"latent":z})["Decoder"]
        zGen = self.Encoder.predict({"image":genImage})["Latent"]
        featuresReal = self.Discriminator.predict({"image":testImages,"features":z})["Features"]
        featuresGen = self.Discriminator.predict({"image":genImage,"features":zGen})["Features"]
        return tf.reduce_sum((featuresReal-featuresGen)**2,axis=-1)

    def AnomalyScore(self,testImages,alpha=0.9):
        v1 = self.ImageAnomaly(testImages)
        v2 = self.FeatureAnomaly(testImages)
        return alpha * v1 + (1-alpha)*v2

class fAnoWGAN(fAnoGAN):

    def __init__(self,settingsDict,dataset,networkConfig={}):
        """Initializing Model and all Hyperparameters """

        self.HPs.update({
                    "DiscrimClipValue":0.01,
                    "GenUpdateFreq":5,
                     })

        super().__init__(settingsDict,dataset,networkConfig)

        self.counter=0

    @tf.function
    def TrainStep(self,images,trainGen=True):
        randomLatent = tf.random.normal([self.HPs["BatchSize"], self.HPs["LatentSize"]])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fakeImage = self.Generator(randomLatent, training=True)["Decoder"]

            e_z = self.Encoder(images)["Latent"]

            out = self.Discriminator({**images,"features":e_z}, training=True)
            realPred = out["Discrim"]
            realFeatures = out["Features"]
            fakePred = self.Discriminator({"image":fakeImage,"features":randomLatent}, training=True)["Discrim"]

            discLoss =  -tf.reduce_mean(realPred) + tf.reduce_mean(fakePred)
            genLoss = -tf.reduce_sum(fakePred)
            encLoss = self.crossEntropy(tf.zeros_like(realPred), realPred)

            # ziz Training
            e_z_fake = self.Encoder(fakeImage)["Latent"]
            zizLoss = self.mse(randomLatent, e_z_fake)

            # izi Training
            reconImage = self.Generator(e_z, training=True)["Decoder"]
            featureFake = self.Discriminator({"image":reconImage,"features":e_z}, training=True)["Features"]
            iziLoss = self.mse(images["image"], tf.squeeze(reconImage))
            iziFeatureLoss = self.mse(realFeatures, featureFake)

            totalLoss = genLoss + encLoss + zizLoss + iziLoss + iziFeatureLoss

        if trainGen:
            generatorGradients = gen_tape.gradient(totalLoss, self.Generator.trainable_variables+self.Encoder.trainable_variables)
            self.generatorOptimizer.apply_gradients(zip(generatorGradients, self.Generator.trainable_variables+self.Encoder.trainable_variables))

        discriminatorGradients = disc_tape.gradient(discLoss, self.Discriminator.trainable_variables)
        self.discriminatorOptimizer.apply_gradients(zip(discriminatorGradients, self.Discriminator.trainable_variables))

        return {"Generator Loss": gen_loss,"Discriminator Loss": disc_loss,"Encoder Loss": enc_loss}

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
        self.ExecuteTrainEndCallbacks()

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
