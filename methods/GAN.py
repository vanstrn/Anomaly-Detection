"""Implementation of Generative Adversarial Networks, first proposed in
https://arxiv.org/abs/1406.2661

If a convolutional network is used then it would be equivalent to
"DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS" (ICLR 2016)
https://arxiv.org/abs/1511.06434
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



class GAN(BaseMethod):
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
                                ]

        #Chacking Valid hyperparameters are specified
        CheckFilled(self.requiredParams,settingsDict["NetworkHPs"])
        self.HPs.update(settingsDict["NetworkHPs"])

        #Processing Other inputs
        self.inputSpec=dataset.inputSpec
        self.opt = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        networkConfig.update(dataset.outputSpec)
        self.Generator = CreateModel(self.HPs["GenNetworkConfig"],{"latent":self.HPs["LatentSize"]},variables=networkConfig)
        self.Discriminator = CreateModel(self.HPs["DiscNetworkConfig"],dataset.inputSpec,variables=networkConfig)

        # self.LoadModel({"modelPath":"models/TestVAE"})
        self.Generator.summary(print_fn=log.info)
        self.Discriminator.summary(print_fn=log.info)

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

    @tf.function
    def train_step(self,images):
        noise = tf.random.normal([self.HPs["BatchSize"], self.HPs["LatentSize"]])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            out = self.Generator(noise, training=True)
            generated_images = {"image":out["Decoder"]}
            real_output = self.Discriminator(images, training=True)["Discrim"]
            fake_output = self.Discriminator(generated_images, training=True)["Discrim"]

            gen_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
            disc_loss = self.cross_entropy(tf.ones_like(real_output), real_output) + \
                        self.cross_entropy(tf.zeros_like(fake_output), fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.Generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.Discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.Generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.Discriminator.trainable_variables))

        return {"Generator Loss": gen_loss,"Discriminator Loss": disc_loss}

    def Test(self,data):
        pass

    def InitializeCallbacks(self,callbacks):
        """Method initializes callbacks for training loops that are not `model.fit()`.
        Additional logic is still required for methods with multiple networks, such as GANs."""
        self.callbacks = tf.keras.callbacks.CallbackList(callbacks,model=self)

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
