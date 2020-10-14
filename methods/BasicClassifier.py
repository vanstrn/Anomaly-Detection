
from utils.utils import CheckFilled
import numpy as np
import tensorflow as tf

class BasicClassifier():
    def __init__(self,settingsDict):
        """Initializing Model and all Hyperparameters """

        self.HPs = {"eps":0.2,
                        "EntropyBeta":0.01,
                        "CriticBeta":1.0,
                        "LearningRate":0.00005,
                        "Optimizer":"Adam",
                        "Epochs":10,
                        "BatchSize":1024,
                        "MinibatchSize":64
                         }

        self.requiredParams = ["NetworkConfig",
                          ]

        #Chacking Valid hyperparameters are specified
        CheckFilled(self.requiredParams,settingsDict["NetworkHPs"])
        self.HPs.update(settingsDict["NetworkHPs"])

        #Processing Other inputs
        self.InitTF1()


    def Train(self,data):
        self.TrainTF1(data)


    def InitTF1(self):

        with tf.device("/cpu:0"):
            #Setting up the model
            from networks.NetworkTF1 import buildNetwork
            netConfigOverride={}
            self.Model = buildNetwork(configFile=self.HPs["NetworkConfig"],netConfigOverride=netConfigOverride)


    def TrainTF1(self,data):
        with tf.device("/cpu:0"):
            if self.HPs["Optimizer"] == "Adam":
                opt = tf.keras.optimizers.Adam(self.HPs["LearningRate"])
            elif self.HPs["Optimizer"] == "RMS":
                opt = tf.keras.optimizers.RMSprop(self.HPs["LearningRate"])
            elif self.HPs["Optimizer"] == "Adagrad":
                opt = tf.keras.optimizers.Adagrad(self.HPs["LearningRate"])
            elif self.HPs["Optimizer"] == "Adadelta":
                opt = tf.keras.optimizers.Adadelta(self.HPs["LearningRate"])
            elif self.HPs["Optimizer"] == "Adamax":
                opt = tf.keras.optimizers.Adamax(self.HPs["LearningRate"])
            elif self.HPs["Optimizer"] == "Nadam":
                opt = tf.keras.optimizers.Nadam(self.HPs["LearningRate"])
            elif self.HPs["Optimizer"] == "SGD":
                opt = tf.keras.optimizers.SGD(self.HPs["LearningRate"])
            elif self.HPs["Optimizer"] == "SGD-Nesterov":
                opt = tf.keras.optimizers.SGD(self.HPs["LearningRate"],nesterov=True)
            elif self.HPs["Optimizer"] == "Amsgrad":
                opt = tf.keras.optimizers.Adam(self.HPs["LearningRate"],amsgrad=True)


            self.Model[0].compile(optimizer=opt, loss=["mse"])

            targets = np.vstack(data[1]).reshape(-1)
            labels = np.eye(10)[targets]

            self.Model[0].fit( [np.expand_dims(np.stack(data[0]),3)],
                            [labels],
                            epochs=self.HPs["Epochs"],
                            batch_size=self.HPs["BatchSize"],
                            shuffle=True,)
