""" Implementation of BiGAN from "Adversarial Feature Learning" (ICLR 2017)
https://arxiv.org/abs/1605.09782

This method is also the same as "Adversarially Learned Inference" (ICLR 2017):
https://arxiv.org/abs/1606.00704

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


class BiGAN(BaseMethod):
    def __init__(self,settingsDict,dataset,networkConfig={}):
        """Initializing Model and all Hyperparameters """

        self.HPs = {
                    "LearningRate":0.00005,
                    "LatentSize":64,
                    "Optimizer":"Adam",
                    "Epochs":10,
                    "BatchSize":32,
                    "Shuffle":True,
                     }

        self.requiredParams = [ "GenNetworkConfig",
                                "DiscNetworkConfig",
                                "EncNetworkConfig",
                                ]

        #Chacking Valid hyperparameters are specified
        CheckFilled(self.requiredParams,settingsDict["NetworkHPs"])
        self.HPs.update(settingsDict["NetworkHPs"])

        #Processing Other inputs
        self.inputSpec=dataset.inputSpec
        self.opt = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        networkConfig.update(dataset.outputSpec)
        networkConfig.update({"LatentSize":self.HPs["LatentSize"]})
        self.Generator = CreateModel(self.HPs["GenNetworkConfig"],{"latent":self.HPs["LatentSize"]},variables=networkConfig)
        self.Encoder = CreateModel(self.HPs["EncNetworkConfig"],dataset.inputSpec,variables=networkConfig)
        _datasetSpec = {"features":[self.HPs["LatentSize"]],**dataset.inputSpec}
        self.Discriminator = CreateModel(self.HPs["DiscNetworkConfig"],_datasetSpec,variables=networkConfig)

        # self.LoadModel({"modelPath":"models/TestVAE"})
        self.Generator.summary(print_fn=log.info)
        self.Discriminator.summary(print_fn=log.info)
        self.Encoder.summary(print_fn=log.info)

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def Train(self,data,callbacks=[]):
        self.InitializeCallbacks(callbacks)
        train_dataset = tf.data.Dataset.from_tensor_slices(data).batch(self.HPs["BatchSize"])
        for epoch in range(self.HPs["Epochs"]):
            ts = time.time()

            for batch in train_dataset:
                info = self.train_step(batch)
            self.ExecuteEpochEndCallbacks(epoch,info)
            print("End Epoch {}: Time {}".format(epoch,time.time()-ts))
        self.ExecuteTrainEndCallbacks({})

    # @tf.function
    def train_step(self,images):
        noise = tf.random.normal([self.HPs["BatchSize"], self.HPs["LatentSize"]])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fakeImage = self.Generator(noise, training=True)["Decoder"]

            e_z = self.Encoder(images)["Latent"]
            real_output = self.Discriminator({**images,"features":e_z}, training=True)["Discrim"]
            fake_output = self.Discriminator({"image":fakeImage,"features":noise}, training=True)["Discrim"]

            disc_loss = self.cross_entropy(tf.ones_like(real_output), real_output) + \
                        self.cross_entropy(tf.zeros_like(fake_output), fake_output)
            gen_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
            enc_loss = self.cross_entropy(tf.zeros_like(real_output), real_output)
            t_loss=gen_loss+enc_loss

        gradients_of_generator = gen_tape.gradient(t_loss, self.Generator.trainable_variables+self.Encoder.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.Discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.Generator.trainable_variables+self.Encoder.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.Discriminator.trainable_variables))

        return {"Generator Loss": gen_loss,"Discriminator Loss": disc_loss,"Encoder Loss": enc_loss}

    def Test(self,data):
        pass

    def InitializeCallbacks(self,callbacks):
        """Method initializes callbacks for training loops that are not `model.fit()`.
        Additional logic is still required for methods with multiple networks, such as GANs."""
        self.callbacks = tf.keras.callbacks.CallbackList(callbacks,model=self)

    def ImagesFromLatent(self,sample):
        return self.Generator.predict(sample)
