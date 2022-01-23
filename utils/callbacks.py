"""
File sets up custom callbacks useful for logging training.
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.math import *
from sklearn import metrics


class LogTraining(tf.keras.callbacks.Callback):
    def __init__(self,logger,dataset):
        super(LogTraining, self).__init__()
        self.logger=logger
        self.dataset=dataset

    def on_epoch_end(self, epoch, logs=None):
        for k,v in logs.items():
            self.logger.LogScalar(tag=k,value=v,step=epoch)


class TestGenerator(tf.keras.callbacks.Callback):
    def __init__(self,logger,dataset,dx=5,dy=5,tbLog=True,rawImage=False,makeGIF=False,fixedLatent=False,name="Generator"):
        super(TestGenerator, self).__init__()
        self.logger=logger
        self.dataset=dataset
        self.dx=dx
        self.dy=dy
        self.rawImage=rawImage
        self.name=name
        self.fixedLatent=fixedLatent
        self.makeGIF=makeGIF
        self.tbLog=tbLog

        self.gifBuffer=[]
        self.latentSample=None

    def on_epoch_end(self, epoch, logs=None):
        """Plotting and saving several test images to specified directory. """

        #Selecting a random subset of images to plot.
        if self.fixedLatent:
            if self.latentSample is None:
                self.latentSample = tf.random.normal([int(self.dx*self.dy), self.model.HPs["LatentSize"]])
            out = self.model.ImagesFromLatent(self.latentSample) # dx*dy X ImageSize
        else:
            latentSample = tf.random.normal([int(self.dx*self.dy), self.model.HPs["LatentSize"]])
            out = self.model.ImagesFromLatent(latentSample) # dx*dy X ImageSize

        x = out.reshape([self.dx,self.dy]+list(out.shape[1:]))
        x = np.squeeze(np.concatenate(np.split(x,self.dx,axis=0),axis=2),axis=0)
        x = np.squeeze(np.concatenate(np.split(x,self.dy,axis=0),axis=2),axis=0)

        if self.tbLog:
            self.logger.LogImage(np.expand_dims(x,0),self.name,epoch)
        if self.rawImage:
            fig = plt.figure()
            if x.shape[-1]==1:
                plt.imshow(x,cmap='Greys')
            plt.axis('off')
            self.logger.SaveImage("{}_Epoch{}".format(self.name,epoch),bbox_inches='tight')

        if self.makeGIF:
            self.gifBuffer.append(NORMtoRGB(x))

    def on_train_end(self, logs=None):
        if self.makeGIF:
            self.logger.SaveGIF(clip=self.gifBuffer,name=self.name,fps=3)


class TestReconstruction(tf.keras.callbacks.Callback):
    def __init__(self,logger,dataset,numTests=4,numTrials=1):
        super(TestReconstruction, self).__init__()
        self.logger=logger
        self.dataset=dataset
        self.numTests=numTests
        self.numTrials=numTrials # Numtrials is used for VAE to visualize different 

    def on_epoch_end(self, epoch, logs=None):
        """Plotting and saving several test images to specified directory. """

        #Selecting a random subset of images to plot.
        imgNums=np.random.randint(self.dataset.testData["image"].shape[0],size=[self.numTests])
        _testImages=self.dataset.testData["image"][imgNums,:]
        testImages = np.repeat(_testImages,self.numTrials,axis=0)

        out = self.model.ImagesFromImage(testImages)
        x = out.reshape([self.numTests,self.numTrials]+list(out.shape[1:]))
        x = np.squeeze(np.concatenate(np.split(x,self.numTests,axis=0),axis=2),axis=0)
        x = np.squeeze(np.concatenate(np.split(x,self.numTrials,axis=0),axis=2),axis=0)

        _testImagesPlot = _testImages.reshape(-1,_testImages.shape[2],_testImages.shape[3])

        finalPlot = np.concatenate([_testImagesPlot,x],axis=1)

        self.logger.LogImage(np.expand_dims(finalPlot,axis=0),"Reconstruction",epoch)


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
