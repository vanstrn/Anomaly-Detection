#!/usr/bin/env python
# Created By  : Neale Van Stralen
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Method definitions for autoencoder-based methods, including variational autoencoders
"""

import tensorflow as tf
import logging

from methods.utils import GetOptimizer,VanishingGradient
from networks.networkKeras import CreateModel
from .BaseMethod import BaseMethod

log = logging.getLogger(__name__)


class Autoencoder(BaseMethod):
    """ Basic training method for autoencoders, which trains to recreate data.
    This method does not use additional changes to improve regularization or overfitting

    For a description of Autoencoders see TBD.

    This version of an Autoencoder implements the Autoencoder in a single network AE(x)==>\hat{x},
    with an intermediate layer of the AE being equivalent to z.
    """

    def __init__(self,settingsDict,dataset,networkConfig={}):
        """Initializing Model and all Hyperparameters """


        self.hyperParams.update({
                    "LearningRate":0.001,
                    "Optimizer":"Adam",
                    "Epochs":10,
                    "BatchSize":64,
                    "GradClipping":None
                     })

        self.requiredParams.Append([
                "NetworkConfig",
                ])

        super().__init__(settingsDict,dataset,networkConfig)

        #Processing Other inputs
        self.inputSpec=dataset.inputSpec
        networkConfig.update(dataset.outputSpec)
        self.Model = CreateModel(self.hyperParams["NetworkConfig"],dataset.inputSpec,variables=networkConfig,printSummary=True)

        self.optimizer = GetOptimizer(self.hyperParams["Optimizer"],self.hyperParams["LearningRate"])
        self.mse = tf.keras.losses.MeanSquaredError()

    @tf.function
    def TrainStep(self,images,hyperParams):

        with tf.GradientTape() as tape:
            generatedImages = self.Model(images, training=True)["Decoder"]

            loss = self.mse(images["image"],generatedImages)

        grads = tape.gradient(loss, self.Model.trainable_variables)
        if hyperParams["GradClipping"] is not None:
                grads, _ = tf.clip_by_global_norm(grads, hyperParams["GradClipping"])
        self.optimizer.apply_gradients([
            (grad, var)
            for (grad,var) in zip(grads, self.Model.trainable_variables)
            if grad is not None])

        return {"Loss/Reconstruction": loss,
            "Gradients/Norm": tf.linalg.global_norm(grads),"Gradients/VanishingRatio": VanishingGradient(grads)}


    def ImagesFromImage(self,testImages):
        return self.Model.predict({"image":testImages},verbose=0)["Decoder"]

    def LatentFromImage(self,testImages):
        return self.Model.predict({"image":testImages},verbose=0)["Latent"]

    def AnomalyScore(self,testImages):
        return tf.reduce_sum((testImages-self.ImagesFromImage(testImages))**2,axis=list(range(1,len(testImages.shape))))


class VariationalAutoencoder(Autoencoder):
    """
    Method specific to Variational Autoencoders, which adds a KL regularization loss to the gaussian latent space.
    """
    def __init__(self,settingsDict,dataset,networkConfig={}):
        """Initializing Model and all Hyperparameters """

        self.hyperParams.update({
            "KLBeta":0.005,
            "GradClipping":1.0,
        })

        super().__init__(settingsDict,dataset,networkConfig)


    @tf.function
    def TrainStep(self,images,hyperParams):

        with tf.GradientTape() as tape:
            data = self.Model(images, training=True)
            generatedImages = data["Decoder"]
            z_mu = data["Mu"]
            z_sigma = data["Sigma"]
            kl_loss = -0.5 * (1 + z_sigma - tf.square(z_mu) - tf.exp(z_sigma))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            reconstruction_loss = self.mse(images["image"],generatedImages)
            loss = reconstruction_loss + hyperParams["KLBeta"] * kl_loss

        grads = tape.gradient(loss, self.Model.trainable_variables)
        if hyperParams["GradClipping"] is not None:
                grads, _ = tf.clip_by_global_norm(grads, hyperParams["GradClipping"])
        self.optimizer.apply_gradients([
            (grad, var)
            for (grad,var) in zip(grads, self.Model.trainable_variables)
            if grad is not None])

        return {"Loss/Total":loss,"Loss/Reconstruction": reconstruction_loss, "Loss/KL Divergence": kl_loss,
            "Gradients/Norm": tf.linalg.global_norm(grads),"Gradients/VanishingRatio": VanishingGradient(grads)}
