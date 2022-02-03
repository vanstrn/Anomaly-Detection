""" Implementation of fAnoGAN from "f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks." (Medical Image Analysis 2019)

Official implementation: https://github.com/tSchlegl/f-AnoGAN
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


class fAnoGAN(BaseMethod):
    def __init__(self,settingsDict,dataset,networkConfig={}):
        """Vanila fAnoGAN implementation which uses two distinct training phases. One for the GAN and one for the encoder."""

        self.HPs.update({
                    "LearningRate":0.00005,
                    "LatentSize":64,
                    "Optimizer":"Adam",
                    "EpochsGAN":10,
                    "EpochsEnc":10,
                    "BatchSize":32,
                    "Shuffle":True,
                    "zizBeta":0.0,
                    "iziBeta":0.0,
                    "iziFeatBeta":1.0,
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
        self.encoderOptimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        self.crossEntropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mse = tf.keras.losses.MeanSquaredError()

    def Train(self,data,callbacks=[]):
        self.InitializeCallbacks(callbacks)
        train_dataset = self.SetupDataset(data)
        for epoch in range(self.HPs["EpochsGAN"]):
            ts = time.time()
            infoList = []
            for batch in train_dataset:
                info = self.TrainStepGenerator(batch)
                infoList.append(info)
            self.ExecuteEpochEndCallbacks(epoch,MergeDictValues(infoList))
            log.info("End Epoch {}: Time {}".format(epoch,time.time()-ts))
        for epoch in range(self.HPs["EpochsEnc"]):
            ts = time.time()
            infoList = []
            for batch in train_dataset:
                info = self.TrainStepEncoder(batch)
                infoList.append(info)
            #Hack to get epoch values to not overlap.
            self.ExecuteEpochEndCallbacks(epoch+self.HPs["EpochsGAN"],MergeDictValues(infoList))
            log.info("End Epoch {}: Time {}".format(epoch,time.time()-ts))
        self.ExecuteTrainEndCallbacks({})

    @tf.function
    def TrainStepGenerator(self,images):
        randomLatent = tf.random.normal([self.HPs["BatchSize"], self.HPs["LatentSize"]])

        with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
            fakeImage = self.Generator(randomLatent, training=True)["Decoder"]

            realPred = self.Discriminator(images, training=True)["Discrim"]
            fakePred = self.Discriminator({"image":fakeImage}, training=True)["Discrim"]

            discLoss =  -tf.reduce_mean(realPred) + tf.reduce_mean(fakePred)
            genLoss = -tf.reduce_sum(fakePred)

        generatorGradients = genTape.gradient(genLoss, self.Generator.trainable_variables)
        self.generatorOptimizer.apply_gradients(zip(generatorGradients, self.Generator.trainable_variables))

        discriminatorGradients = discTape.gradient(discLoss, self.Discriminator.trainable_variables)
        self.discriminatorOptimizer.apply_gradients(zip(discriminatorGradients, self.Discriminator.trainable_variables))

        return {"Generator Loss": genLoss,"Discriminator Loss": discLoss}

    @tf.function
    def TrainStepEncoder(self,images):

        with tf.GradientTape() as encTape:
            totalLoss = 0.0
            # ziz Training
            if self.HPs["zizBeta"] != 0.0:
                randomLatent = tf.random.normal([self.HPs["BatchSize"], self.HPs["LatentSize"]])
                fakeImage = self.Generator(randomLatent, training=True)["Decoder"]
                e_z_fake = self.Encoder(fakeImage)["Latent"]
                zizLoss = self.mse(randomLatent, e_z_fake)
            else:
                zizLoss=0.0
            # izi Training
            if self.HPs["iziBeta"] != 0.0:
                e_z = self.Encoder(images)["Latent"]
                reconImage = self.Generator(e_z, training=True)["Decoder"]
                realFeatures = self.Discriminator(images, training=True)["Features"]
                featureFake = self.Discriminator({"image":reconImage}, training=True)["Features"]
                iziLoss = self.mse(images["image"], reconImage)
                iziFeatureLoss = self.mse(realFeatures, featureFake)
            else:
                iziLoss = 0.0
                iziFeatureLoss = 0.0


            totalLoss = zizLoss*self.HPs["zizBeta"] + iziLoss*self.HPs["iziBeta"] + iziFeatureLoss*self.HPs["iziFeatBeta"]

        encoderGradients = encTape.gradient(totalLoss, self.Encoder.trainable_variables)
        self.encoderOptimizer.apply_gradients(zip(encoderGradients, self.Encoder.trainable_variables))

        return {"ZIZ Loss": zizLoss,"IZI Loss": iziLoss,"IZI Feature Loss": iziFeatureLoss}

    def ImagesFromLatent(self,sample):
        return self.Generator.predict(sample)["Decoder"]

    def LatentFromImage(self,sample):
        return self.Encoder.predict(sample)["Latent"]

    def ImagesFromImage(self,testImages):
        z = self.Encoder.predict({"image":testImages})["Latent"]
        return self.Generator.predict({"latent":z})["Decoder"]

    def DiscrimAnomaly(self,testImages):
        z = self.Encoder.predict({"image":testImages})["Latent"]
        return self.Discriminator.predict({"image":testImages,"features":z})["Discrim"]

    def ImageAnomaly(self,testImages):
        return tf.reduce_sum((testImages-self.ImagesFromImage(testImages))**2,axis=list(range(1,len(testImages.shape))))

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


class fAnoGAN_v2(BiGAN):
    def __init__(self,*args,**kwargs):
        """fAnoGAN custom implementation which uses a vanila GAN (instead of Wasserstein).
        This method also implements the training in one step updating the encoder with the generator.
        Finally this method also uses a BiGAN discriminator where the discriminator is a function of the latent representation i.e. D=f(x,z) """
        self.HPs.update({
                    "zizBeta":0.0,
                    "iziBeta":1.0,
                    "iziFeatBeta":1.0,
                     })
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

            # ziz Training
            e_z_fake = self.Encoder(generatedImages)["Latent"]
            zizLoss = self.mse(randomLatent, e_z_fake)

            # izi Training
            reconImage = self.Generator(e_z, training=True)["Decoder"]
            fakeFeatures = self.Discriminator({"image":reconImage,"features":e_z}, training=True)["Features"]
            iziLoss = self.mse(images["image"], reconImage)
            iziFeatureLoss = self.mse(realFeatures, fakeFeatures)

            totalloss = genLoss + zizLoss*self.HPs["zizBeta"] + iziLoss*self.HPs["iziBeta"] + iziFeatureLoss*self.HPs["iziFeatBeta"]

        generatorGradients = genTape.gradient(totalloss, self.Generator.trainable_variables+self.Encoder.trainable_variables)
        discriminatorGradients = discTape.gradient(discLoss, self.Discriminator.trainable_variables)

        self.generatorOptimizer.apply_gradients(zip(generatorGradients, self.Generator.trainable_variables+self.Encoder.trainable_variables))
        self.discriminatorOptimizer.apply_gradients(zip(discriminatorGradients, self.Discriminator.trainable_variables))

        return {"Generator Loss": genLoss,"Discriminator Loss": discLoss,"Encoder Loss": encLoss}

    def DiscrimAnomaly(self,testImages):
        z = self.Encoder.predict({"image":testImages})["Latent"]
        return self.Discriminator.predict({"image":testImages,"features":z})["Discrim"]

    def ImageAnomaly(self,testImages):
        return tf.reduce_sum((testImages-self.ImagesFromImage(testImages))**2,axis=list(range(1,len(testImages.shape))))

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

class fAnoWGAN_v2(fAnoGAN):

    def __init__(self,settingsDict,dataset,networkConfig={}):
        """fAnoGAN custom implementation which uses a Wasserstein GAN.
        This method also implements the training in one step updating the encoder with the generator.
        This method also uses discriminator clipping and modifications to generator update frequency, similar to vanila WGAN methods.
        Finally this method also uses a BiGAN discriminator where the discriminator is a function of the latent representation i.e. D=f(x,z)
        """

        self.HPs.update({
                    "DiscrimClipValue":0.01,
                    "GenUpdateFreq":5,
                    "zizBeta":0.0,
                    "iziBeta":1.0,
                    "iziFeatBeta":1.0,
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

            # ziz Training
            e_z_fake = self.Encoder(fakeImage)["Latent"]
            zizLoss = self.mse(randomLatent, e_z_fake)

            # izi Training
            reconImage = self.Generator(e_z, training=True)["Decoder"]
            featureFake = self.Discriminator({"image":reconImage,"features":e_z}, training=True)["Features"]
            iziLoss = self.mse(images["image"], reconImage)
            iziFeatureLoss = self.mse(realFeatures, featureFake)

            totalLoss = genLoss + zizLoss*self.HPs["zizBeta"] + iziLoss*self.HPs["iziBeta"] + iziFeatureLoss*self.HPs["iziFeatBeta"]

        if trainGen:
            generatorGradients = gen_tape.gradient(totalLoss, self.Generator.trainable_variables+self.Encoder.trainable_variables)
            self.generatorOptimizer.apply_gradients(zip(generatorGradients, self.Generator.trainable_variables+self.Encoder.trainable_variables))

        discriminatorGradients = disc_tape.gradient(discLoss, self.Discriminator.trainable_variables)
        self.discriminatorOptimizer.apply_gradients(zip(discriminatorGradients, self.Discriminator.trainable_variables))

        return {"Generator Loss": genLoss,"Discriminator Loss": discLoss,"Encoder Loss": encLoss}

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
