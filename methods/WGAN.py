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

        self.generator_optimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        self.discriminator_optimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])

        self.counter=0

    def Train(self,data,callbacks=[]):
        self.InitializeCallbacks(callbacks)
        train_dataset = tf.data.Dataset.from_tensor_slices(data).batch(self.HPs["BatchSize"])
        for epoch in range(self.HPs["Epochs"]):
            ts = time.time()

            for batch in train_dataset:
                trainGen = (self.counter % self.HPs["GenUpdateFreq"] == 0)
                info = self.train_step(batch,trainGen)
                self.ClipDiscriminator()
                self.counter+=1
            self.ExecuteEpochEndCallbacks(epoch,info)
            print("End Epoch {}: Time {}".format(epoch,time.time()-ts))
        self.ExecuteTrainEndCallbacks({})

    @tf.function
    def train_step(self,images,trainGen=True):
        noise = tf.random.normal([self.HPs["BatchSize"], self.HPs["LatentSize"]])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            out = self.Generator(noise, training=True)
            generated_images = {"image":out["Decoder"]}
            real_output = self.Discriminator(images, training=True)["Discrim"]
            fake_output = self.Discriminator(generated_images, training=True)["Discrim"]

            gen_loss = -tf.reduce_sum(fake_output)
            disc_loss =  -tf.reduce_mean(real_output) + tf.reduce_mean(fake_output)

        if trainGen:
            gradients_of_generator = gen_tape.gradient(gen_loss, self.Generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.Generator.trainable_variables))

        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.Discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.Discriminator.trainable_variables))

        return {"Generator Loss": gen_loss,"Discriminator Loss": disc_loss}

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
        return self.Generator.predict(sample)
