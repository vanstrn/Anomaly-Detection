"""Implementation of Generative Adversarial Networks, first proposed in
https://arxiv.org/abs/1406.2661

If a convolutional network is used then it would be equivalent to
"DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS" (ICLR 2016)
https://arxiv.org/abs/1511.06434
 """
import tensorflow as tf
import logging

from methods.utils import GetOptimizer
from networks.networkKeras import CreateModel
from .BaseMethod import BaseMethod

log = logging.getLogger(__name__)


class GAN(BaseMethod):
    def __init__(self,settingsDict,dataset,networkConfig={}):
        """Initializing Models for the method """

        self.HPs.update({
                    "LearningRate":0.0001,
                    "LatentSize":64,
                    "Optimizer":"Adam",
                    "Epochs":10,
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
        self.crossEntropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    @tf.function
    def TrainStep(self,images):
        randomLatent = tf.random.normal([self.HPs["BatchSize"], self.HPs["LatentSize"]])

        with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
            generatedImages = self.Generator(randomLatent, training=True)["Decoder"]

            realPred = self.Discriminator(images, training=True)["Discrim"]
            fakePred = self.Discriminator({"image":generatedImages}, training=True)["Discrim"]

            genLoss = self.crossEntropy(tf.ones_like(fakePred), fakePred)
            discLoss = self.crossEntropy(tf.ones_like(realPred), realPred) + \
                        self.crossEntropy(tf.zeros_like(fakePred), fakePred)

        generatorGradients = genTape.gradient(genLoss, self.Generator.trainable_variables)
        discriminatorGradients = discTape.gradient(discLoss, self.Discriminator.trainable_variables)

        self.generatorOptimizer.apply_gradients(zip(generatorGradients, self.Generator.trainable_variables))
        self.discriminatorOptimizer.apply_gradients(zip(discriminatorGradients, self.Discriminator.trainable_variables))

        return {"Generator Loss": genLoss,"Discriminator Loss": discLoss}

    def ImagesFromLatent(self,sample):
        return self.Generator.predict(sample)["Decoder"]
