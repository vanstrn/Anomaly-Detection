
import tensorflow as tf
import logging

from methods.utils import GetOptimizer
from networks.networkKeras import CreateModel
from .BaseMethod import BaseMethod

log = logging.getLogger(__name__)


class Autoencoder(BaseMethod):
    def __init__(self,settingsDict,dataset,networkConfig={}):
        """Initializing Model and all Hyperparameters """

        self.HPs.update({
                    "LearningRate":0.00005,
                    "Optimizer":"Adam",
                    "Epochs":10,
                    "BatchSize":64,
                    "Shuffle":True,
                     })

        self.requiredParams.Append(["NetworkConfig",
                          ])

        super().__init__(settingsDict)

        #Processing Other inputs
        self.opt = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        networkConfig.update(dataset.outputSpec)
        self.Model = CreateModel(self.HPs["NetworkConfig"],dataset.inputSpec,variables=networkConfig,printSummary=True)
        self.Model.compile(optimizer=self.opt, loss=["mse"],metrics=[])

    def Train(self,data,callbacks=[]):
        self.InitializeCallbacks(callbacks)
        self.Model.fit( data["image"],
                        data["image"],
                        epochs=self.HPs["Epochs"],
                        batch_size=self.HPs["BatchSize"],
                        shuffle=self.HPs["Shuffle"],
                        callbacks=self.callbacks)
        self.SaveModel("models/TestAE")

    def ImagesFromImage(self,testImages):
        return self.Model.predict({"image":testImages})["Decoder"]

    def AnomalyScore(self,testImages):
        return tf.reduce_sum((testImages-self.ImagesFromImage(testImages))**2,axis=list(range(1,len(testImages.shape))))
