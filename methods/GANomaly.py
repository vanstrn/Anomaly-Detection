
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

        self.generator_optimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        self.discriminator_optimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        self.encoder_optimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def Train(self,data,callbacks=[]):
        self.InitializeCallbacks(callbacks)
        train_dataset = self.SetupDataset(data)
        for epoch in range(self.HPs["Epochs"]):
            ts = time.time()

            for batch in train_dataset:
                info1={}
                info2 = self.train_step_XZX(batch)
            self.ExecuteEpochEndCallbacks(epoch,{**info1,**info2})
            log.info("End Epoch {}: Time {}".format(epoch,time.time()-ts))
        self.ExecuteTrainEndCallbacks({})

    @tf.function
    def train_step_GAN(self,images):
        noise = tf.random.normal([self.HPs["BatchSize"], self.HPs["LatentSize"]])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            out = self.Generator(noise, training=True)
            generated_images = {"image":out["Decoder"]}
            real_output = self.Discriminator(images, training=True)
            fake_output = self.Discriminator(generated_images, training=True)
            real_label = real_output["Discrim"]; real_features = real_output["Features"]
            fake_label = fake_output["Discrim"]; fake_features = fake_output["Features"]

            gen_loss = self.cross_entropy(tf.ones_like(fake_label), fake_label)
            # feat_loss = tf.reduce_mean(real_features-fake_features)**2.0
            disc_loss = self.cross_entropy(tf.ones_like(real_label), real_label) + \
                        self.cross_entropy(tf.zeros_like(fake_label), fake_label)
            gt_loss = gen_loss # + feat_loss

        gradients_of_generator = gen_tape.gradient(gt_loss, self.Generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.Discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.Generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.Discriminator.trainable_variables))

        return {"Generator Loss": gen_loss,"Discriminator Loss": disc_loss}

    # @tf.function
    def train_step_XZX(self,images):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as enc_tape, tf.GradientTape() as disc_tape:
            z = self.Encoder(images, training=True)["Latent"]
            x_hat = self.Generator(z, training=True)["Decoder"]
            z_hat = self.Encoder2(x_hat, training=True)["Latent"]

            real_output = self.Discriminator(images, training=True)
            fake_output = self.Discriminator(x_hat, training=True)
            real_label = real_output["Discrim"]; real_features = real_output["Features"]
            fake_label = fake_output["Discrim"]; fake_features = fake_output["Features"]

            gen_loss = self.cross_entropy(tf.ones_like(fake_label), fake_label)
            feat_loss = tf.reduce_mean(real_features-fake_features)**2.0
            disc_loss = self.cross_entropy(tf.ones_like(real_label), real_label) + \
                        self.cross_entropy(tf.zeros_like(fake_label), fake_label)

            l_encoder = (z_hat-z)**2
            l_context = tf.math.abs(tf.squeeze(x_hat)-images["image"])
            total_loss = tf.reduce_mean(l_encoder) +tf.reduce_mean(l_context) + gen_loss + feat_loss

        gradients_of_generator = gen_tape.gradient(total_loss, self.Generator.trainable_variables)
        gradients_of_encoder = enc_tape.gradient(total_loss, self.Encoder.trainable_variables+self.Encoder2.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(total_loss, self.Discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.Generator.trainable_variables))
        self.encoder_optimizer.apply_gradients(zip(gradients_of_encoder, self.Encoder.trainable_variables+self.Encoder2.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.Discriminator.trainable_variables))

        return {"Encoding Loss": tf.reduce_mean(l_encoder),"Construction Loss": tf.reduce_mean(l_context)}

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
        return self.Generator.predict(sample)

    def ImagesFromImage(self,testImages):
        z = self.Encoder.predict({"image":testImages})["Latent"]
        return self.Generator.predict({"latent":z})["Decoder"]

    def AnomalyScore(self,testImages):
        return tf.reduce_sum((testImages-tf.squeeze(self.ImagesFromImage(testImages)))**2,axis=[1,2])
