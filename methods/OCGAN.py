""" Implementation of OCGAN from "OCGAN: One-class Novelty Detection Using GANs with Constrained Latent
Representations" (CVPR 2019)
https://openaccess.thecvf.com/content_CVPR_2019/papers/Perera_OCGAN_One-Class_Novelty_Detection_Using_GANs_With_Constrained_Latent_Representations_CVPR_2019_paper.pdf

Code implementations:
https://github.com/PramuPerera/OCGAN
https://github.com/nuclearboy95/Anomaly-Detection-OCGAN-tensorflow

Method Description: Uses denoising autoencoder

Method Comments:
The classifier is redundant to the image discriminator.
"""
import numpy as np
import tensorflow as tf
import logging
import time

from utils.utils import MergeDictValues
from methods.utils import GetOptimizer
from networks.networkKeras import CreateModel
from .BaseMethod import BaseMethod

log = logging.getLogger(__name__)


class OCGAN(BaseMethod):
    def __init__(self,settingsDict,dataset,networkConfig={}):
        """Initializing Model and all Hyperparameters """

        self.HPs.update({
                    "LearningRate":0.00005,
                    "LatentSize":64,
                    "Optimizer":"Adam",
                    "Epochs":10,
                    "BatchSize":32,
                    "Shuffle":True,
                    "NoiseMagnitude":0.2
                     })

        self.requiredParams.Append([
            "GenNetworkConfig",
            "DiscImageNetworkConfig",
            "DiscLatentNetworkConfig",
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
        _datasetSpec = {**dataset.inputSpec}
        self.DiscriminatorImage = CreateModel(self.HPs["DiscImageNetworkConfig"],_datasetSpec,variables=networkConfig,printSummary=True)
        self.Classifier = CreateModel(self.HPs["DiscImageNetworkConfig"],_datasetSpec,variables=networkConfig,printSummary=True)
        _datasetSpec = {"latent":[self.HPs["LatentSize"]]}
        self.DiscriminatorLatent = CreateModel(self.HPs["DiscLatentNetworkConfig"],_datasetSpec,variables=networkConfig,printSummary=True)

        self.generatorOptimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        self.discriminatorOptimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        self.classifierOptimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        self.crossEntropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    @tf.function
    def TrainStep(self,images,hyperParams):
        randomLatent = tf.random.uniform([self.HPs["BatchSize"], self.HPs["LatentSize"]],minval=-1,maxval=1)
        imageNoise = tf.random.normal(images["image"].shape,stddev=self.HPs["NoiseMagnitude"])


        with tf.GradientTape() as genTape, tf.GradientTape() as discTape, tf.GradientTape() as classTape:
            generatedImages1 = self.Generator(randomLatent, training=True)["Decoder"]
            e_z = self.Encoder(images["image"]+imageNoise)["Latent"]
            generatedImages2 = self.Generator(e_z, training=True)["Decoder"]

            realPred = self.DiscriminatorImage({**images}, training=True)["Discrim"]
            fakePred = self.DiscriminatorImage({"image":generatedImages2}, training=True)["Discrim"]

            realLatentPred = self.DiscriminatorLatent({"latent":randomLatent}, training=True)["Discrim"]
            fakeLatentPred = self.DiscriminatorLatent({"latent":e_z}, training=True)["Discrim"]

            realClassifier = self.Classifier({"image":generatedImages1}, training=True)["Discrim"]
            fakeClassifier = self.Classifier({"image":generatedImages2}, training=True)["Discrim"]

            classLoss = self.crossEntropy(tf.ones_like(realClassifier), realClassifier) + \
                        self.crossEntropy(tf.zeros_like(fakeClassifier), fakeClassifier)
            discLoss =  self.crossEntropy(tf.ones_like(realPred), realPred) + \
                        self.crossEntropy(tf.zeros_like(fakePred), fakePred) + \
                        self.crossEntropy(tf.ones_like(realLatentPred), realLatentPred) + \
                        self.crossEntropy(tf.zeros_like(fakeLatentPred), fakeLatentPred)
            genLoss =   self.crossEntropy(tf.ones_like(fakePred), fakePred) + \
                        self.crossEntropy(tf.zeros_like(realPred), realPred) + \
                        self.crossEntropy(tf.ones_like(fakeLatentPred), fakeLatentPred) + \
                        self.crossEntropy(tf.zeros_like(realLatentPred), realLatentPred)

            mseLoss = tf.reduce_mean(images["image"]-generatedImages2)**2.0


        generatorGradients = genTape.gradient(genAllLoss, self.Generator.trainable_variables+self.Encoder.trainable_variables)
        discriminatorGradients = discTape.gradient(discLoss, self.DiscriminatorImage.trainable_variables+self.DiscriminatorLatent.trainable_variables)
        classifierGradients = classTape.gradient(classLoss, self.Classifier.trainable_variables)

        self.generatorOptimizer.apply_gradients(zip(generatorGradients, self.Generator.trainable_variables+self.Encoder.trainable_variables))
        self.discriminatorOptimizer.apply_gradients(zip(discriminatorGradients, self.DiscriminatorImage.trainable_variables+self.DiscriminatorLatent.trainable_variables))
        self.classifierOptimizer.apply_gradients(zip(classifierGradients, self.Classifier.trainable_variables))

        return {"Generator Loss": genLoss,"Discriminator Loss": discLoss,"MSE Loss": mseLoss, "Classifier Loss": classLoss}

    def ImagesFromLatent(self,sample):
        return self.Generator.predict(sample)["Decoder"]

    def LatentFromImage(self,sample):
        return self.Encoder.predict(sample)["Latent"]

    def ImagesFromImage(self,testImages):
        z = self.Encoder.predict({"image":testImages})["Latent"]
        return self.Generator.predict({"latent":z})["Decoder"]

    def ImageDiscrim(self,testImages):
        return self.DiscriminatorImage.predict({"image":testImages})["Discrim"]

    def AnomalyScore(self,testImages,alpha=0.9):
        v1 = tf.reduce_sum((testImages-self.ImagesFromImage(testImages))**2,axis=list(range(1,len(testImages.shape))))
        v2 = tf.squeeze(self.ImageDiscrim(testImages))
        return alpha * v1 + (1-alpha)*v2

class OCGAN_v2(OCGAN):
    def __init__(self,settingsDict,dataset,networkConfig={}):
        """Initializing Model and all Hyperparameters """

        self.HPs.update({
                    "InformNegUpdates":0,
                    "InformNegWait":15,
                    "OptimizerLatent":"SGD",
                    "LearningRateLatent":0.1,
                    "Lambda":500.,
                     })
        super().__init__(settingsDict,dataset,networkConfig)
        self.randomLatent = tf.Variable(tf.random.uniform([self.HPs["BatchSize"], self.HPs["LatentSize"]],minval=-1,maxval=1))
        self.latentOptimizer = GetOptimizer(self.HPs["OptimizerLatent"],self.HPs["LearningRateLatent"])

        self.discriminatorZOptimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        self.encoderOptimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])

    def Train(self,data,callbacks=[]):
        self.InitializeCallbacks(callbacks)
        train_dataset = self.SetupDataset(data)
        for epoch in range(self.HPs["Epochs"]):
            ts = time.time()
            infoList = []
            for batch in train_dataset:
                info = self.TrainStep(batch,epoch)
                infoList.append(info)
            self.ExecuteEpochEndCallbacks(epoch,MergeDictValues(infoList))
            log.info("End Epoch {}: Time {}".format(epoch,time.time()-ts))
        self.ExecuteTrainEndCallbacks({})

    def SetupDataset(self,data):
        # imageNoise = tf.random.normal(data["image"].shape,stddev=self.HPs["NoiseMagnitude"])
        data["imageNoisy"] = data["image"] + tf.random.normal(data["image"].shape,stddev=self.HPs["NoiseMagnitude"])
        dataset = tf.data.Dataset.from_tensor_slices(data)
        if self.HPs["ShuffleData"]:
            dataset = dataset.shuffle(self.HPs["ShuffleSize"])
        return dataset.batch(self.HPs["BatchSize"],self.HPs["DropRemainder"])

    @tf.function
    def TrainStep(self,images,epoch):

        self.randomLatent.assign(tf.random.uniform([self.HPs["BatchSize"], self.HPs["LatentSize"]],minval=-1,maxval=1))


        with tf.GradientTape() as discXTape, tf.GradientTape() as discZTape, tf.GradientTape() as classTape:
            generatedImages1 = self.Generator(self.randomLatent, training=True)["Decoder"]
            e_z = self.Encoder(images["imageNoisy"])["Latent"]
            generatedImages2 = self.Generator(e_z, training=True)["Decoder"]

            realPred = self.DiscriminatorImage({"image":images["image"]}, training=True)["Discrim"]
            fakePred = self.DiscriminatorImage({"image":generatedImages2}, training=True)["Discrim"]

            realLatentPred = self.DiscriminatorLatent({"latent":self.randomLatent}, training=True)["Discrim"]
            fakeLatentPred = self.DiscriminatorLatent({"latent":e_z}, training=True)["Discrim"]

            realClassifier = self.Classifier({"image":generatedImages2}, training=True)["Discrim"]
            fakeClassifier = self.Classifier({"image":generatedImages1}, training=True)["Discrim"]

            classLoss = self.crossEntropy(tf.ones_like(realClassifier), realClassifier) + \
                        self.crossEntropy(tf.zeros_like(fakeClassifier), fakeClassifier)

            discLossX = self.crossEntropy(tf.ones_like(realPred), realPred) + \
                        self.crossEntropy(tf.zeros_like(fakePred), fakePred)

            discLossZ = self.crossEntropy(tf.ones_like(realLatentPred), realLatentPred) + \
                        self.crossEntropy(tf.zeros_like(fakeLatentPred), fakeLatentPred)
            discLoss = discLossX + discLossZ

        discriminatorXGradients = discXTape.gradient(discLossX, self.DiscriminatorImage.trainable_variables)
        discriminatorZGradients = discZTape.gradient(discLossZ, self.DiscriminatorLatent.trainable_variables)
        classifierGradients = classTape.gradient(classLoss, self.Classifier.trainable_variables)

        self.discriminatorOptimizer.apply_gradients(zip(discriminatorXGradients, self.DiscriminatorImage.trainable_variables))
        self.discriminatorZOptimizer.apply_gradients(zip(discriminatorZGradients, self.DiscriminatorLatent.trainable_variables))
        self.classifierOptimizer.apply_gradients(zip(classifierGradients, self.Classifier.trainable_variables))

        #Informative-negative mining
        # if epoch >= self.HPs["InformNegWait"]:
        # if (epoch == 0) or (epoch >= self.HPs["InformNegWait"]):
        for i in range(self.HPs["InformNegUpdates"]):
            with tf.GradientTape() as latentTape:
                generatedImages1 = self.Generator(self.randomLatent, training=True)["Decoder"]
                realClassifier = self.Classifier({"image":generatedImages1}, training=True)["Discrim"]
                latentLoss = self.crossEntropy(tf.ones_like(realClassifier), realClassifier)
            latentGradients = latentTape.gradient(latentLoss, [self.randomLatent])
            self.latentOptimizer.apply_gradients(zip(latentGradients, [self.randomLatent]))
        self.randomLatent.assign(tf.clip_by_value(self.randomLatent,-1.,1.))

        with tf.GradientTape() as genTape:
            e_z = self.Encoder(images["imageNoisy"])["Latent"]
            generatedImages1 = self.Generator(self.randomLatent, training=True)["Decoder"]
            generatedImages2 = self.Generator(e_z, training=True)["Decoder"]

            # realPred = self.DiscriminatorImage({"image":images["image"]}, training=True)["Discrim"]
            fakePred = self.DiscriminatorImage({"image":generatedImages1}, training=True)["Discrim"]

            # realLatentPred = self.DiscriminatorLatent({"latent":self.randomLatent}, training=True)["Discrim"]
            fakeLatentPred = self.DiscriminatorLatent({"latent":e_z}, training=True)["Discrim"]

            genLoss =   self.crossEntropy(tf.ones_like(fakePred), fakePred)
            encLoss =   self.crossEntropy(tf.ones_like(fakeLatentPred), fakeLatentPred)
            mseLoss = self.HPs["Lambda"] * tf.reduce_mean(images["image"]-generatedImages2)**2.0
            genAllLoss = genLoss + mseLoss
            encAllLoss = encLoss + mseLoss

        generatorGradients = genTape.gradient(genAllLoss, self.Generator.trainable_variables)
        self.generatorOptimizer.apply_gradients(zip(generatorGradients, self.Generator.trainable_variables))
        encoderGradients = genTape.gradient(encAllLoss, self.Encoder.trainable_variables)
        self.encoderOptimizer.apply_gradients(zip(encoderGradients, self.Encoder.trainable_variables))

        return {"Generator Loss": genLoss,"Discriminator X Loss": discLossX, "Discriminator Z Loss": discLossZ,"MSE Loss": mseLoss, "Classifier Loss": classLoss, "Encoder Loss": encLoss}
