"""Deep One-Class Classification (DSVDD), first proposed in
http://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf

#TO DO:
Add weight decay terms to the training.
 """
import tensorflow as tf
import logging
import time
import numpy as np

from methods.utils import GetOptimizer
from networks.networkKeras import CreateModel
from .BaseMethod import BaseMethod
from utils.utils import MergeDictValues


log = logging.getLogger(__name__)


class DSVDD(BaseMethod):
    def __init__(self,settingsDict,dataset,networkConfig={}):
        """Initializing Models for the method """

        self.HPs.update({
                    "LearningRate":0.0001,
                    "LatentSize":32,
                    "Optimizer":"Adam",
                    "PretrainEpochs":1,
                    "TrainEpochs":25,
                    "CenterEpsilon":0.1,
                    "Nu":0.1,
                    "Objective":"OneClass",
                    "WeightDecay":1e-6,
                    "UpdateRadiusK":1,
                     })

        self.requiredParams.Append([
                "GenNetworkConfig",
                "EncNetworkConfig",
                ])

        super().__init__(settingsDict)

        #Processing Other inputs
        self.inputSpec=dataset.inputSpec
        networkConfig.update(dataset.outputSpec)
        self.Generator = CreateModel(self.HPs["GenNetworkConfig"],{"latent":self.HPs["LatentSize"]},variables=networkConfig,printSummary=True)
        self.Encoder = CreateModel(self.HPs["EncNetworkConfig"],dataset.inputSpec,variables=networkConfig,printSummary=True)

        self.optimizer = GetOptimizer(self.HPs["Optimizer"],self.HPs["LearningRate"])
        self.crossEntropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        self.mse = tf.keras.losses.MeanSquaredError()
        self.variables = self.Generator.trainable_variables + self.Encoder.trainable_variables

        self.Radius = tf.Variable(0)

    def Train(self,data,callbacks=[]):
        self.InitializeCallbacks(callbacks)
        trainDataset = self.SetupDataset(data)
        for epoch in range(self.HPs["PretrainEpochs"]):
            ts = time.time()
            infoList = []
            for batch in trainDataset:
                info = self.PreTrainStep(batch)
                infoList.append(info)
            # self.ExecuteEpochEndCallbacks(epoch,MergeDictValues(infoList))
            log.info("End Pretrain Epoch {}/{}: Time {}".format(epoch+1,self.HPs["PretrainEpochs"],time.time()-ts))

        self.InitCenter(trainDataset)

        for epoch in range(self.HPs["TrainEpochs"]):
            ts = time.time()
            infoList = []
            if epoch % self.HPs["UpdateRadiusK"] == 0:
                distances=[]
                for batch in trainDataset:
                    distances.append(self.GetRadius(batch))
                self.UpdateRadius(tf.concat(distances,axis=0))
            # print("Radius",self.Radius)
            # print(dir(self.Radius))
            # print("Radius",self.Radius.eval())
            for batch in trainDataset:
                info = self.TrainStep(batch)
                infoList.append(info)
            self.ExecuteEpochEndCallbacks(epoch,MergeDictValues(infoList))
            log.info("End Epoch {}: Time {}".format(epoch,time.time()-ts))
        self.ExecuteTrainEndCallbacks({})

    @tf.function
    def PreTrainStep(self,images):
        """Pretraining of an Autoencoder which generates a good representation for the Encoder network."""

        with tf.GradientTape() as gradTape:

            latent = self.Encoder(images, training=True)["Latent"]
            generatedImages = self.Generator(latent, training=True)["Decoder"]

            loss = self.mse(images["image"],generatedImages)

        gradients = gradTape.gradient(loss, self.variables)

        self.optimizer.apply_gradients(zip(gradients, self.variables))

        return {"Autoencoder Loss": loss,}

    @tf.function
    def TrainStep(self,images,hyperParams):

        with tf.GradientTape() as gradTape:
            latent = self.Encoder(images, training=True)["Latent"]
            dist = tf.reduce_sum((latent - self.center)**2,axis=-1)

            if self.HPs["Objective"] == "OneClass":
                loss = tf.reduce_mean(dist)
            if self.HPs["Objective"] == "SoftBoundary":
                scores = dist - self.Radius ** 2
                loss = self.Radius ** 2 + (1 / self.HPs["Nu"]) * tf.reduce_mean(tf.reduce_max(tf.zeros_like(scores), scores))

        gradients = gradTape.gradient(loss, self.Encoder.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.Encoder.trainable_variables))

        return {"Radius Loss": loss, "Distance":tf.reduce_mean(dist)}

    @tf.function
    def GetRadius(self,images):
        latent = self.Encoder(images, training=True)["Latent"]
        dist = tf.reduce_sum((latent - self.center)**2,axis=-1)
        return dist

    @tf.function
    def UpdateRadius(self,dist):
        self.Radius = tf.numpy_function(GetRadius,[dist,self.HPs["Nu"]],tf.float32)

    def InitCenter(self, dataSet):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        nSamples = 0
        center = tf.zeros([self.HPs["LatentSize"]])

        for data in dataSet:
            # get the inputs of the batch
            latent = self.Encoder(data)["Latent"]
            nSamples += latent.shape[0]
            center += tf.reduce_sum(latent, axis=0)

        center /= nSamples

        # If center_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        lt = tf.where((center > -self.HPs["CenterEpsilon"]) & (center < 0))
        center = tf.where((center > -self.HPs["CenterEpsilon"]) & (center < 0), -self.HPs["CenterEpsilon"], center)
        center = tf.where((center <  self.HPs["CenterEpsilon"]) & (center > 0),  self.HPs["CenterEpsilon"], center)

        self.center = center

    def AnomalyScore(self,testImages,alpha=0.9):
        latent = self.Encoder.predict(testImages)["Latent"]
        return tf.reduce_mean((latent - self.center)**2,axis=-1)

    def LatentFromImage(self,sample):
        return self.Encoder.predict(sample)["Latent"]


def GetRadius(dist, nu):
    """Simple minibatch solution for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist), 1 - nu).astype(np.float32)



def BlockRUpdate(rep, center, nu, solver='minimize_scalar', scalar_method='brent', lp_obj='primal', tol=0.001, **kwargs):
    """
    Function to update R while leaving the network parameters and center c fixed in a block coordinate optimization.
    Using scipy.optimize.minimize_scalar or linear programming of cvxopt.
    solver: should be either "minimize_scalar" (default) or "lp" (linear program)
    scalar_method: the optimization method used in minimize_scalar ('brent', 'bounded', or 'golden')
    lp_obj: should be either "primal" (default) or "dual"
    """

    assert solver in ("minimize_scalar", "lp")

    if solver == "minimize_scalar":

        from scipy.optimize import minimize_scalar

        assert scalar_method in ("brent", "bounded", "golden")


        n, d = rep.shape
        dist = np.sum((rep - center) ** 2, axis=1, dtype=np.float32)

        # define deep SVDD objective function in R
        def f(x):
            return (x**2 + (1 / (nu * n)) *
                    np.sum(np.max(np.column_stack((np.zeros(n), dist - x**2)), axis=1), dtype=np.float32))

        # get lower and upper bounds around the (1-nu)-th quantile of distances
        bracket = None
        bounds = None

        upper_idx = int(np.max((np.floor(n * nu * 0.1), 1)))
        lower_idx = int(np.min((np.floor(n * nu * 1.1), n)))
        sort_idx = dist.argsort()
        upper = dist[sort_idx][-upper_idx]
        lower = dist[sort_idx][-lower_idx]

        if scalar_method in ("brent", "golden"):
            bracket = (lower, upper)

        elif scalar_method == "bounded":
            bounds = (lower, upper)

        # solve for R
        res = minimize_scalar(f, bracket=bracket, bounds=bounds, method=scalar_method)

        # Get new R
        R = res.x

    elif solver == "lp":

        from cvxopt import matrix
        from cvxopt.solvers import lp

        assert lp_obj in ("primal", "dual")


        n, d = rep.shape

        # Solve either primal or dual objective
        if lp_obj == "primal":

            # Define LP
            c = matrix(np.append(np.ones(1), (1 / ( nu * n)) * np.ones(n), axis=0).astype(np.double))
            G = matrix(- np.concatenate((np.concatenate((np.ones(n).reshape(n,1), np.eye(n)), axis=1),
                                         np.concatenate((np.zeros(n).reshape(n, 1), np.eye(n)), axis=1)),
                                        axis=0).astype(np.double))
            h = matrix(np.append(- np.sum((rep - center) ** 2, axis=1), np.zeros(n), axis=0).astype(np.double))

            # Solve LP
            sol = lp(c, G, h)['x']

            # Get new R
            R = np.array(sol).reshape(n+1).astype(np.float32)[0]

        elif lp_obj == "dual":

            # Define LP
            c = matrix((np.sum((rep - center) ** 2, axis=1)).astype(np.double))
            G = matrix((np.concatenate((np.eye(n), -np.eye(n)), axis=0)).astype(np.double))
            h = matrix((np.concatenate(((1/( nu*n)) * np.ones(n), np.zeros(n)), axis=0)).astype(np.double))
            A = matrix((np.ones(n)).astype(np.double)).trans()
            b = matrix(1, tc='d')

            # Solve LP
            sol = lp(c, G, h, A, b)['x']
            a = np.array(sol).reshape(n)

            # Recover R (using the specified numeric tolerance on the range)
            n_svs = 0  # number of support vectors
            while n_svs == 0:
                lower = tol * (1/( nu*n))
                upper = (1-tol) * (1/( nu*n))
                idx_svs = (a > lower) & (a < upper)
                n_svs = np.sum(idx_svs)
                tol /= 10  # decrease tolerance if there are still no support vectors found

            R = np.median(np.array(c).reshape(n)[idx_svs]).astype(np.float32)

    return R
