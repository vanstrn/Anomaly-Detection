
import numpy as np
import tensorflow as tf
import logging

from utils.utils import CheckFilled
from methods.utils import GetOptimizer
from networks.networkKeras import CreateModel
from .BaseMethod import BaseMethod

log = logging.getLogger(__name__)

class Classifier(BaseMethod):
    def __init__(self,settingsDict,dataset,networkConfig={}):
        """Initializing Model and all Hyperparameters """

        self.HPs = {
                    "LearningRate":0.00005,
                    "Optimizer":"Adam",
                    "Epochs":10,
                    "BatchSize":64,
                    "Shuffle":True,
                     }

        self.requiredParams = ["NetworkConfig",
                          ]

        #Chacking Valid hyperparameters are specified
        CheckFilled(self.requiredParams,settingsDict["HyperParams"])
        self.HPs.update(settingsDict["HyperParams"])

        #Processing Other inputs
        self.opt = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        networkConfig.update(dataset.outputSpec)
        self.Model = CreateModel(self.HPs["NetworkConfig"],dataset.inputSpec,variables=networkConfig)
        self.Model.compile(optimizer=self.opt, loss=["mse"],metrics=['accuracy'])
        self.Model.summary(print_fn=log.info)

        self.LoadModel({"modelPath":"models/Test"})

    def Train(self,data,callbacks=[]):

        self.Model.fit( data["x_train"],
                        data["y_train"],
                        epochs=self.HPs["Epochs"],
                        batch_size=self.HPs["BatchSize"],
                        shuffle=self.HPs["Shuffle"],
                        callbacks=callbacks)
        self.SaveModel("models/Test")

    def Test(self,data):
        count = 0
        for i in range(0,data["x_test"].shape[0],self.HPs["BatchSize"]):
            if i + self.HPs["BatchSize"] > data["x_test"].shape[0]:
                pred = self.Model(np.expand_dims(data["x_test"][i:,:,:],-1))
                realFinal = tf.math.argmax(data["y_test"][i:,:],axis=-1)
            else:
                pred = self.Model(np.expand_dims(data["x_test"][i:i+self.HPs["BatchSize"],:,:],-1))
                realFinal = tf.math.argmax(data["y_test"][i:i+self.HPs["BatchSize"],:],axis=-1)
            predFinal = tf.math.argmax(pred['Classifier'],axis=-1)
            count += tf.reduce_sum(tf.cast(tf.math.equal(realFinal,predFinal),dtype=tf.float32))
        print(count,data["x_test"].shape[0])
        print(count/data["x_test"].shape[0])
