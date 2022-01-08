
import numpy as np
import tensorflow as tf
import logging
import matplotlib.pyplot as plt

from utils.utils import CheckFilled
from methods.utils import GetOptimizer
from networks.networkKeras import CreateModel
from .BaseMethod import BaseMethod
from sklearn import metrics
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


class Autoencoder(BaseMethod):
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
        CheckFilled(self.requiredParams,settingsDict["NetworkHPs"])
        self.HPs.update(settingsDict["NetworkHPs"])

        #Processing Other inputs
        self.opt = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        networkConfig.update(dataset.outputSpec)
        self.Model = CreateModel(self.HPs["NetworkConfig"],dataset.inputSpec,variables=networkConfig)
        self.Model.compile(optimizer=self.opt, loss=["mse"],metrics=[])

        # self.LoadModel({"modelPath":"models/TestVAE"})
        self.Model.summary(print_fn=log.info)

    def Train(self,data,callbacks=[]):
        self.InitializeCallbacks(callbacks)
        self.Model.fit( data["image"],
                        data["image"],
                        epochs=self.HPs["Epochs"],
                        batch_size=self.HPs["BatchSize"],
                        shuffle=self.HPs["Shuffle"],
                        callbacks=self.callbacks)
        self.SaveModel("models/TestAE")

    def Test(self,data):
        pass

    def ImagesFromImage(self,testImages):
        return self.Model.predict({"image":testImages})["Decoder"]

    def AnomalyScore(self,testImages):
        return tf.reduce_sum((testImages-tf.squeeze(self.ImagesFromImage(testImages)))**2,axis=[1,2])


class TestReconstruction(tf.keras.callbacks.Callback):
    def __init__(self,logger,dataset,numTests=4,numTrials=1):
        super(TestReconstruction, self).__init__()
        self.logger=logger
        self.dataset=dataset
        self.numTests=numTests
        self.numTrials=numTrials

    def on_epoch_end(self, epoch, logs=None):
        """Plotting and saving several test images to specified directory. """

        #Selecting a random subset of images to plot.
        imgNums=np.random.randint(self.dataset.testData["image"].shape[0],size=[self.numTests])
        _testImages=self.dataset.testData["image"][imgNums,:]
        testImages = np.repeat(_testImages,self.numTrials,axis=0)
        #
        out = self.model.ImagesFromImage(testImages)
        x = out.reshape([self.numTests,self.numTrials]+list(out.shape[1:]))
        x2 = np.concatenate(np.split(x,self.numTests,axis=0),axis=2)
        x3 = np.concatenate(np.split(x2,self.numTrials,axis=1),axis=3)

        _testImages2 = _testImages.reshape(-1,28)

        finalPlot = np.concatenate([np.squeeze(_testImages2),np.squeeze(x3)],axis=1)

        self.logger.LogImage(np.expand_dims(finalPlot,axis=(0,-1)),"Reconstruction",epoch)

class TestAnomaly(tf.keras.callbacks.Callback):
    def __init__(self,logger,dataset):
        super(TestAnomaly, self).__init__()
        self.logger=logger
        self.dataset=dataset

    def on_epoch_end(self, epoch, logs=None):
        """Plotting and saving several test images to specified directory. """

        #Selecting a random subset of images to plot.
        _testImages=self.dataset.testData["image"]
        #
        anom_score = self.model.AnomalyScore(_testImages)
        labels=self.dataset.testData["anom_label"]
        fpr, tpr, thresholds = metrics.roc_curve(labels,anom_score)
        roc_auc = metrics.auc(fpr, tpr)

        fig = plt.figure(figsize=(5,5))
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        self.logger.LogMatplotLib(fig,"ROC_Curve",epoch)
        self.logger.LogScalar("AUC",roc_auc,epoch)
