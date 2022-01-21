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

    @tf.function
    def train_step(self,images):
        noise = tf.random.normal([self.HPs["BatchSize"], self.HPs["LatentSize"]])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fakeImage = self.Generator(noise, training=True)["Decoder"]

            e_z = self.Encoder(images)["Latent"]

            out= self.Discriminator({**images,"features":e_z}, training=True)
            real_output = out["Discrim"]
            featureReal = out["Features"]
            fake_output = self.Discriminator({"image":fakeImage,"features":noise}, training=True)["Discrim"]

            disc_loss = self.cross_entropy(tf.ones_like(real_output), real_output) + \
                        self.cross_entropy(tf.zeros_like(fake_output), fake_output)
            gen_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
            enc_loss = self.cross_entropy(tf.zeros_like(real_output), real_output)

            mse = tf.keras.losses.MeanSquaredError()
            # ziz Training
            e_z_fake = self.Encoder(fakeImage)["Latent"]
            ziz_loss = mse(noise, e_z_fake)

            # izi Training
            reconImage = self.Generator(e_z, training=True)["Decoder"]
            featureFake = self.Discriminator({"image":reconImage,"features":e_z}, training=True)["Features"]
            izi_loss = mse(images["image"], tf.squeeze(reconImage))
            izi_f_loss = mse(featureReal, featureFake)

            t_loss = gen_loss + enc_loss + ziz_loss + izi_loss + izi_f_loss

        gradients_of_generator = gen_tape.gradient(t_loss, self.Generator.trainable_variables+self.Encoder.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.Discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.Generator.trainable_variables+self.Encoder.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.Discriminator.trainable_variables))

        return {"Generator Loss": gen_loss,"Discriminator Loss": disc_loss,"Encoder Loss": enc_loss}

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
    def train_step(self,images,trainGen=True):
        noise = tf.random.normal([self.HPs["BatchSize"], self.HPs["LatentSize"]])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fakeImage = self.Generator(noise, training=True)["Decoder"]

            e_z = self.Encoder(images)["Latent"]

            out= self.Discriminator({**images,"features":e_z}, training=True)
            real_output = out["Discrim"]
            featureReal = out["Features"]
            fake_output = self.Discriminator({"image":fakeImage,"features":noise}, training=True)["Discrim"]

            disc_loss =  -tf.reduce_mean(real_output) + tf.reduce_mean(fake_output)
            gen_loss = -tf.reduce_sum(fake_output)
            enc_loss = self.cross_entropy(tf.zeros_like(real_output), real_output)

            mse = tf.keras.losses.MeanSquaredError()
            # ziz Training
            e_z_fake = self.Encoder(fakeImage)["Latent"]
            ziz_loss = mse(noise, e_z_fake)

            # izi Training
            reconImage = self.Generator(e_z, training=True)["Decoder"]
            featureFake = self.Discriminator({"image":reconImage,"features":e_z}, training=True)["Features"]
            izi_loss = mse(images["image"], tf.squeeze(reconImage))
            izi_f_loss = mse(featureReal, featureFake)

            t_loss = gen_loss + enc_loss #+ ziz_loss + izi_loss + izi_f_loss
        if trainGen:
            gradients_of_generator = gen_tape.gradient(t_loss, self.Generator.trainable_variables+self.Encoder.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.Generator.trainable_variables+self.Encoder.trainable_variables))

        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.Discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.Discriminator.trainable_variables))

        return {"Generator Loss": gen_loss,"Discriminator Loss": disc_loss,"Encoder Loss": enc_loss}

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
