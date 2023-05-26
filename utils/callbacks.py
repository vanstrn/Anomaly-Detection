"""
File sets up custom callbacks useful for logging training.
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.math import *
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns


class LogTraining(tf.keras.callbacks.Callback):
    def __init__(self,logger,dataset):
        super(LogTraining, self).__init__()
        self.logger=logger
        self.dataset=dataset

    def on_epoch_end(self, epoch, logs=None):
        for k,v in logs.items():
            self.logger.LogScalar(tag=k,value=v,step=epoch)


class PlotGenerator(tf.keras.callbacks.Callback):
    def __init__(self,logger,dataset,dx=5,dy=5,tbLog=True,rawImage=False,makeGIF=False,fixedLatent=False,name="Generation",fps=3):
        super(PlotGenerator, self).__init__()
        self.logger=logger
        self.dataset=dataset
        self.dx=dx
        self.dy=dy
        self.rawImage=rawImage
        self.name=name
        self.fixedLatent=fixedLatent
        self.makeGIF=makeGIF
        self.tbLog=tbLog
        self.fps=fps

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
                plt.imshow(x,cmap='Greys_r')
            plt.axis('off')
            self.logger.SaveImage(fig,"{}_Epoch{}".format(self.name,epoch),bbox_inches='tight')

        plt.close()
        if self.makeGIF:
            self.gifBuffer.append(NORMtoRGB(x))

    def on_train_end(self, logs=None):
        if self.makeGIF:
            self.logger.SaveGIF(clip=self.gifBuffer,name=self.name,fps=self.fps)


class PlotReconstruction(tf.keras.callbacks.Callback):
    def __init__(self,logger,dataset,numTests=4,numTrials=1,dy=1,rawImage=False,fixedSet=False,makeGIF=False,name="Reconstruction"):
        super(PlotReconstruction, self).__init__()
        self.logger=logger
        self.dataset=dataset
        self.numTests=numTests
        self.numTrials=numTrials # Numtrials is used for VAE to visualize different
        self.rawImage=rawImage
        self.fixedSet=fixedSet
        self.dy=dy
        self.makeGIF=makeGIF
        self.name=name
        self.set=None
        self.gifBuffer=[]

    def on_epoch_end(self, epoch, logs=None):
        """Plotting and saving several test images to specified directory. """

        #Selecting a random subset of images to plot.
        imageColumns=[]
        for i in range(self.dy):
            if self.fixedSet:
                if self.set is None:
                    self.set=[np.random.randint(self.dataset.testData["image"].shape[0],size=[self.numTests]) for j in range(self.dy)]
                imgNums=self.set[i]
            else:
                imgNums=np.random.randint(self.dataset.testData["image"].shape[0],size=[self.numTests])
            _testImages=self.dataset.testData["image"][imgNums,:]
            testImages = np.repeat(_testImages,self.numTrials,axis=0)

            out = self.model.ImagesFromImage(testImages)
            x = out.reshape([self.numTests,self.numTrials]+list(out.shape[1:]))
            x = np.squeeze(np.concatenate(np.split(x,self.numTests,axis=0),axis=2),axis=0)
            x = np.squeeze(np.concatenate(np.split(x,self.numTrials,axis=0),axis=2),axis=0)

            _testImagesPlot = _testImages.reshape(-1,_testImages.shape[2],_testImages.shape[3])

            columnPlot = np.concatenate([_testImagesPlot,x],axis=1)
            imageColumns.append(columnPlot)
            if i != self.dy-1: #Adding column division
                imageColumns.append(np.ones(shape=[columnPlot.shape[0],1,columnPlot.shape[2]]))

        finalPlot = np.concatenate(imageColumns,axis=1)
        self.logger.LogImage(np.expand_dims(finalPlot,axis=0),"Reconstruction",epoch)
        if self.rawImage:
            fig = plt.figure()
            plt.imshow(finalPlot)
            self.logger.SaveImage(fig,"{}_Epoch{}".format("Reconstruction",epoch),bbox_inches='tight')
        plt.close()

        if self.makeGIF:
            self.gifBuffer.append(NORMtoRGB(finalPlot))

    def on_train_end(self, logs=None):
        if self.makeGIF:
            self.logger.SaveGIF(clip=self.gifBuffer,name=self.name,fps=3)


class PlotImageAnomaly(tf.keras.callbacks.Callback):
    def __init__(self,logger,dataset,dxNorm=1,dxAnom=1,dy=5,rawImage=False,fixedSet=False,makeGIF=False):
        super(PlotImageAnomaly, self).__init__()
        self.logger=logger
        self.dataset=dataset
        self.dxNorm=dxNorm
        self.dxAnom=dxAnom
        self.dy=dy
        self.rawImage=rawImage
        self.fixedSet=fixedSet
        self.makeGIF=makeGIF
        self.set=None
        self.gifBuffer=[]

    def on_epoch_end(self, epoch, logs=None):
        """Plotting and saving several test images to specified directory. """

        #Selecting a random subset of images to plot.
        if self.fixedSet:
            if self.set is None:
                self.normSet=np.random.choice(np.where(np.squeeze(self.dataset.testData["anom_label"])==0)[0],size=self.dy*self.dxNorm)
                self.anomSet=np.random.choice(np.where(np.squeeze(self.dataset.testData["anom_label"])==1)[0],size=self.dy*self.dxAnom)
            imgNums = np.concatenate([self.normSet,self.anomSet])
        else:
            normSet=np.random.choice(np.where(np.squeeze(self.dataset.testData["anom_label"])==0)[0],size=self.dy*self.dxNorm)
            anomSet=np.random.choice(np.where(np.squeeze(self.dataset.testData["anom_label"])==1)[0],size=self.dy*self.dxAnom)
            imgNums = np.concatenate([normSet,anomSet])

        testImages=self.dataset.testData["image"][imgNums,:]
        _testImages = testImages.reshape([self.dxNorm+self.dxAnom,self.dy,]+list(testImages.shape[1:]))
        _testImages = np.squeeze(np.concatenate(np.split(_testImages,self.dxNorm+self.dxAnom,axis=0),axis=3),axis=0)
        _testImages = np.squeeze(np.concatenate(np.split(_testImages,self.dy,axis=0),axis=1),axis=0)

        out = self.model.ImagesFromImage(testImages)
        x = out.reshape([self.dxNorm+self.dxAnom,self.dy,]+list(out.shape[1:]))
        x = np.squeeze(np.concatenate(np.split(x,self.dxNorm+self.dxAnom,axis=0),axis=3),axis=0)
        x = np.squeeze(np.concatenate(np.split(x,self.dy,axis=0),axis=1),axis=0)

        anomalyMap = (_testImages-x)**2
        if anomalyMap.shape[-1] > 1:
            anomalyMap = np.mean(anomalyMap,axis=-1)

        fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
        fig.suptitle('Anomaly Detection (L - Normal Images | R - Anomalous Images)')
        if x.shape[-1]==1:
            ax1.imshow(_testImages,cmap='Greys_r');ax1.set_title('Input Image');ax1.axis('off')
            ax2.imshow(x,cmap='Greys_r');ax2.set_title('Predicted Image');ax2.axis('off')
        else:
            ax1.imshow(NORMtoRGB(_testImages));ax1.set_title('Input Image');ax1.axis('off')
            ax2.imshow(NORMtoRGB(x));ax2.set_title('Predicted Image');ax2.axis('off')
        ax3.imshow(anomalyMap);ax3.set_title('Anomaly Difference');ax3.axis('off')

        if self.rawImage:
            self.logger.SaveImage(fig,"{}_Epoch{}".format("Anomaly",epoch),bbox_inches='tight')
        self.logger.LogMatplotLib(fig,"Anomaly Detection",epoch)
        plt.close()

        if self.makeGIF:
            pass

    def on_train_end(self, logs=None):
        if self.makeGIF:
            pass
            # self.logger.SaveGIF(clip=self.gifBuffer,name=self.name,fps=3)


class TestAnomaly(tf.keras.callbacks.Callback):
    def __init__(self,logger,dataset,rawImage=False):
        super(TestAnomaly, self).__init__()
        self.logger=logger
        self.dataset=dataset
        self.rawImage=rawImage

    def on_epoch_end(self, epoch, logs=None):
        """Plotting and saving several test images to specified directory. """

        anom_score = self.model.AnomalyScore(self.dataset.testData["image"])
        labels=self.dataset.testData["anom_label"]
        fpr, tpr, thresholds = metrics.roc_curve(labels,anom_score)
        roc_auc = metrics.auc(fpr, tpr)

        fig = plt.figure(figsize=(5,5))
        plt.title('Receiver Operating Characteristic (Test Data)')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        self.logger.LogMatplotLib(fig,"ROC_Curve",epoch)
        self.logger.LogScalar("AUC/Test",roc_auc,epoch)
        if self.rawImage:
            self.logger.SaveImage(fig,"{}_Epoch{}".format("ROC_Curve_Test",epoch),bbox_inches='tight')
        plt.close()


class ValidationAnomaly(tf.keras.callbacks.Callback):
    def __init__(self,logger,dataset,rawImage=False):
        super(ValidationAnomaly, self).__init__()
        self.logger=logger
        self.dataset=dataset
        self.rawImage=rawImage

    def on_epoch_end(self, epoch, logs=None):
        """Plotting and saving several test images to specified directory. """

        anom_score = self.model.AnomalyScore(self.dataset.validationData["image"])
        labels=self.dataset.validationData["anom_label"]
        fpr, tpr, thresholds = metrics.roc_curve(labels,anom_score)
        roc_auc = metrics.auc(fpr, tpr)

        fig = plt.figure(figsize=(5,5))
        plt.title('Receiver Operating Characteristic (Validation Data)')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        self.logger.LogMatplotLib(fig,"ROC_Curve",epoch)
        self.logger.LogScalar("AUC/Validation",roc_auc,epoch)
        if self.rawImage:
            self.logger.SaveImage(fig,"{}_Epoch{}".format("ROC_Curve_Validation",epoch),bbox_inches='tight')
        plt.close()


class PlotLatentSpace(tf.keras.callbacks.Callback):
    def __init__(self,logger,dataset,rawImage=False,method="TSNE"):
        super(PlotLatentSpace, self).__init__()
        self.logger=logger
        self.dataset=dataset
        self.rawImage=rawImage
        if method == "PCA":
            self.method=PCA
        else:
            self.method=TSNE


    def on_epoch_end(self, epoch, logs=None):
        """Plotting and saving several test images to specified directory. """

        latent = self.model.LatentFromImage(self.dataset.testData["image"])
        labels = self.dataset.testData["label"]


        feat_cols = [ 'latent'+str(i) for i in range(latent.shape[1]) ]
        df = pd.DataFrame(latent,columns=feat_cols)
        df['y'] = labels
        df['label'] = df['y'].apply(lambda i: str(i))
        data = self.method(n_components=2)
        data_result = data.fit_transform(df[feat_cols].values)
        df['dim-one'] = data_result[:,0]
        df['dim-two'] = data_result[:,1]

        labelSet = sorted(set(df['label']))

        fig = plt.figure(figsize=(7,7))
        plt.title('Latent Visualization')
        sns.scatterplot(
        x="dim-one", y="dim-two",
        data=df,
        hue_order = labelSet,
        # palette="label",
        hue="label",
        legend="full",
        alpha=0.7
        )

        self.logger.LogMatplotLib(fig,"Latent_Vis",epoch)
        if self.rawImage:
            self.logger.SaveImage(fig,"{}_Epoch{}".format("Latent_Vis",epoch),bbox_inches='tight')
        plt.close()

class PlotLatentReconstruction(tf.keras.callbacks.Callback):
    def __init__(self,logger,dataset,yDim=10,xDim=10,method="PCA",rawImage=False,name="LatentReconstruction"):
        super(PlotLatentReconstruction, self).__init__()
        self.logger=logger
        self.dataset=dataset
        self.rawImage=rawImage
        self.name=name
        self.xDim=xDim
        self.yDim=yDim
        if method == "PCA":
            self.decompMethod=PCA
        else:
            self.decompMethod=TSNE

    def on_epoch_end(self, epoch, logs=None):
        """Plotting a 2D sweep over the latent space to see latent space utilization in the real image space.
        If the dimension of latent space is greater than 2, then 2D directions are selected based on the principal components of the test dataset."""

        try:
            latentSize = model.latentSize
        except:
            latentSize=3

        if latentSize ==2:
            latentFactor = np.array([[1,0],0,1])
        else:
            _testImages=self.dataset.testData["image"]
            #
            latent = self.model.LatentFromImage(_testImages)

            feat_cols = [ 'latent'+str(i) for i in range(latent.shape[1]) ]
            latentData = pd.DataFrame(latent,columns=feat_cols)
            decompInstance = self.decompMethod(n_components=2)
            latentDataTransformed = decompInstance.fit_transform(latentData[feat_cols].values)
            maxLatentTransformed = np.amax(latentDataTransformed,axis=0)
            minLatentTransformed = np.amin(latentDataTransformed,axis=0)
            latentFactor=decompInstance.components_

        xArray = np.linspace(maxLatentTransformed[0]+0.1,minLatentTransformed[0]-0.1,self.xDim)
        yArray = np.linspace(maxLatentTransformed[1]+0.1,minLatentTransformed[1]-0.1,self.yDim)
        latentTransformedTestCombos = np.stack(np.meshgrid(xArray,yArray), -1).reshape(-1, 2)
        latentTestCombos = np.matmul(latentTransformedTestCombos,latentFactor)

        latentImages = self.model.ImagesFromLatent(latentTestCombos)
        latentImagesReshape = latentImages.reshape([self.xDim,self.yDim]+list(latentImages.shape[1:]))
        latentImagesReshape = np.squeeze(np.concatenate(np.split(latentImagesReshape,self.xDim,axis=0),axis=2),axis=0)
        finalPlot = np.squeeze(np.concatenate(np.split(latentImagesReshape,self.yDim,axis=0),axis=2),axis=0)

        self.logger.LogImage(np.expand_dims(finalPlot,axis=0),"LatentReconstruction",epoch)
        if self.rawImage:
            fig = plt.figure()
            plt.imshow(finalPlot)
            self.logger.SaveImage(fig,"{}_Epoch{}".format("LatentReconstruction",epoch),bbox_inches='tight')
        plt.close()

    #     if self.makeGIF:
    #         self.gifBuffer.append(NORMtoRGB(finalPlot))
    #
    # def on_train_end(self, logs=None):
    #     if self.makeGIF:
    #         self.logger.SaveGIF(clip=self.gifBuffer,name=self.name,fps=3)
