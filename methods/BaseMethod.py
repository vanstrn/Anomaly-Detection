import tensorflow as tf
import logging
from methods.utils import Requirements
import time
from pprint import pformat
from utils.utils import MergeDictValues
from methods.schedulers import GetScheduler

log = logging.getLogger(__name__)

class BaseMethod():
    hyperParams = { "BatchSize":32,
            "ShuffleData":True,
            "ShuffleSize":10000,
            "DropRemainder":False,
            }
    requiredParams = Requirements()

    def __init__(self, settingsDict, *args, **kwargs):
        #Initializing
        #Chacking Valid hyperparameters are specified
        self.requiredParams.Check(settingsDict["HyperParams"])
        self.hyperParams.update(settingsDict["HyperParams"])
        if "Schedulers" in self.hyperParams:
            self.InitializeSchedulers()

        log.info("Hyperparameters:\n{}".format(pformat(self.hyperParams)))


    def SetupDataset(self,data):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        if self.hyperParams["ShuffleData"]:
            dataset = dataset.shuffle(self.hyperParams["ShuffleSize"])
        return dataset.batch(self.hyperParams["BatchSize"],self.hyperParams["DropRemainder"])

    def Train(self,data,callbacks=[]):
        self.InitializeCallbacks(callbacks)
        train_dataset = self.SetupDataset(data)
        for epoch in range(self.hyperParams["Epochs"]):
            ts = time.time()
            infoList = []
            for batch in train_dataset:
                info = self.TrainStep(batch,hyperParams=self.hyperParams)
                infoList.append(info)
            self.ExecuteEpochEndCallbacks(epoch,MergeDictValues(infoList))
            log.info("End Epoch {}: Time {}".format(epoch,time.time()-ts))

            if "Schedulers" in self.hyperParams:
                self.UpdateSchedulers(episode=epoch,**MergeDictValues(infoList))
        self.ExecuteTrainEndCallbacks({})

    def Test(self):
        self.ExecuteTrainEndCallbacks({})

    def SaveModel(self,modelPath):
        """Saves model in specified folder. Assumes """
        self.Model.save(modelPath)
        # self.Model.save(modelPath+"/saved_model.h5")
        log.info("Saved model to {}".format(modelPath))

    def LoadModel(self,settings):
        """Loads specified models.  """
        modelPath = settings["modelPath"]
        try:
            self.Model = tf.keras.models.load_model(modelPath)
            log.info("Loaded Keras model from {} ".format(modelPath))
        except OSError:
            log.info("Could not find Keras model from {} ".format(modelPath))
        except:
            log.warning("Failed to Load Keras Model from {}".format(modelPath))

    def ExecuteEpochEndCallbacks(self,epoch,logs=None):
        self.callbacks.on_epoch_end(epoch,logs)

    def ExecuteTrainEndCallbacks(self,logs=None):
        self.callbacks.on_train_end(logs)

    def InitializeCallbacks(self,callbacks):
        """Method initializes callbacks for training loops that are not `model.fit()`.
        Pass any params that the callbacks need into the generation of the callback list.

        For methods with multiple networks, pass them is as dictionaries.
        This requires callbacks that are compatible with the dictionary style of model usage.
        This style is compatible with the `method.fit()` method b/c code nests the inputed model variable without performing checks.
        Future it might be desirable to create a custom model nesting logic that will allow callbacks like `ModelCheckpoint` to be compatible.
        """
        self.callbacks = tf.keras.callbacks.CallbackList(callbacks,model=self)

    def load_weights(self):
        pass

    def save_weights(self):
        pass

    def set_weights(self):
        pass

    def InitializeSchedulers(self):
        self.schedulers={}
        for schedulerInfo in self.hyperParams["Schedulers"]:
            schedName=schedulerInfo.pop("Variable")
            self.schedulers[schedName] = GetScheduler(**schedulerInfo)
        for varName, varScheduler in self.schedulers.items():
            self.hyperParams[varName] = tf.Variable(varScheduler.StepValue(episode=0))

    def UpdateSchedulers(self,episode=0,**kwargs):
        for varName, varScheduler in self.schedulers.items():
            self.hyperParams[varName].assign(varScheduler.StepValue(episode=episode,**kwargs))


if __name__ == "__main__":
    testMethod = BaseMethod()
    print(dir(BaseMethod))
    print(BaseMethod.hyperParams)
