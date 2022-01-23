"""
File sets up custom callbacks useful for logging training.
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.math import *

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

        x = out["Decoder"].reshape([self.dx,self.dy]+list(out["Decoder"].shape[1:]))
        # Stacking images together to create array of images.
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
