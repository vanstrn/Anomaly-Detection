"""Implementation of Generative Adversarial Networks, first proposed in
https://arxiv.org/abs/1406.2661

If a convolutional network is used then it would be equivalent to
"DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS" (ICLR 2016)
https://arxiv.org/abs/1511.06434
 """
import numpy as np
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

        self.generator_optimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        self.discriminator_optimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def train_step(self,images):
        noise = tf.random.normal([self.HPs["BatchSize"], self.HPs["LatentSize"]])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generatedImages = self.Generator(noise, training=True)["Decoder"]

            real_output = self.Discriminator(images, training=True)["Discrim"]
            fake_output = self.Discriminator({"image":generatedImages}, training=True)["Discrim"]

            gen_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
            disc_loss = self.cross_entropy(tf.ones_like(real_output), real_output) + \
                        self.cross_entropy(tf.zeros_like(fake_output), fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.Generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.Discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.Generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.Discriminator.trainable_variables))

        return {"Generator Loss": gen_loss,"Discriminator Loss": disc_loss}

    def ImagesFromLatent(self,sample):
        return self.Generator.predict(sample)


class TestGenerator(tf.keras.callbacks.Callback):
    def __init__(self,logger,dataset,dx=5,dy=5):
        super(TestGenerator, self).__init__()
        self.logger=logger
        self.dataset=dataset
        self.dx=dx
        self.dy=dy

    def on_epoch_end(self, epoch, logs=None):
        """Plotting and saving several test images to specified directory. """

        #Selecting a random subset of images to plot.
        latentSample = tf.random.normal([int(self.dx*self.dy), self.model.HPs["LatentSize"]])
        #
        out = self.model.ImagesFromLatent(latentSample)
        x = out["Decoder"].reshape([self.dx,self.dy]+list(out["Decoder"].shape[1:]))
        x2 = np.concatenate(np.split(x,self.dx,axis=0),axis=2)
        x3 = np.squeeze(np.concatenate(np.split(x2,self.dy,axis=1),axis=3))


        self.logger.LogImage(np.expand_dims(x3,axis=(0,-1)),"Generator",epoch)
