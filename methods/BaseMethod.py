import tensorflow as tf
import logging
from methods.utils import Requirements
import time
from pprint import pformat
from utils.utils import MergeDictValues

log = logging.getLogger(__name__)

class BaseMethod():
    HPs = { "BatchSize":32,
            "ShuffleData":True,
            "ShuffleSize":10000,
            "DropRemainder":False,
            }
    requiredParams = Requirements()

    def __init__(self, settingsDict, *args, **kwargs):
        #Initializing
        #Chacking Valid hyperparameters are specified
        self.requiredParams.Check(settingsDict["NetworkHPs"])
        self.HPs.update(settingsDict["NetworkHPs"])

        log.info("Hyperparameters:\n{}".format(pformat(self.HPs)))


    def SetupDataset(self,data):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        if self.HPs["ShuffleData"]:
            dataset = dataset.shuffle(self.HPs["ShuffleSize"])
        return dataset.batch(self.HPs["BatchSize"],self.HPs["DropRemainder"])

    def Train(self,data,callbacks=[]):
        self.InitializeCallbacks(callbacks)
        train_dataset = self.SetupDataset(data)
        for epoch in range(self.HPs["Epochs"]):
            ts = time.time()
            infoList = []
            for batch in train_dataset:
                info = self.TrainStep(batch)
                infoList.append(info)
            self.ExecuteEpochEndCallbacks(epoch,MergeDictValues(infoList))
            log.info("End Epoch {}: Time {}".format(epoch,time.time()-ts))
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

if __name__ == "__main__":
    testMethod = BaseMethod()
    print(dir(BaseMethod))
    print(BaseMethod.HPs)
