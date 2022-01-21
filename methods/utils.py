import tensorflow as tf

from utils.utils import GetFunction
import logging

log = logging.getLogger(__name__)

def LoadMethod(settingDict,dataset,**kwargs):
    Method = GetFunction(settingDict["Method"])
    net = Method(settingDict,dataset,**kwargs)
    return net

def GetOptimizer(optimizer,learningRate):
    if optimizer == "Adam":
        opt = tf.keras.optimizers.Adam(learningRate)
    elif optimizer == "RMS":
        opt = tf.keras.optimizers.RMSprop(learningRate)
    elif optimizer == "Adagrad":
        opt = tf.keras.optimizers.Adagrad(learningRate)
    elif optimizer == "Adadelta":
        opt = tf.keras.optimizers.Adadelta(learningRate)
    elif optimizer == "Adamax":
        opt = tf.keras.optimizers.Adamax(learningRate)
    elif optimizer == "Nadam":
        opt = tf.keras.optimizers.Nadam(learningRate)
    elif optimizer == "SGD":
        opt = tf.keras.optimizers.SGD(learningRate)
    elif optimizer == "SGD-Nesterov":
        opt = tf.keras.optimizers.SGD(learningRate,nesterov=True)
    elif optimizer == "Amsgrad":
        opt = tf.keras.optimizers.Adam(learningRate,amsgrad=True)
    else:
        raise Exception("Invalid optimizer `{}` specified.".format(optimizer))

    return opt

class Requirements():
    def __init__(self):
        self.requirements=set()

    def Append(self,requirementList):
        for requirement in requirementList:
            self.requirements.add(requirement)

    def Check(self,dictionary,fileName=None):
        valid = True
        for requiredParam in self.requirements:
            if requiredParam not in dictionary:
                valid = False
                if fileName is not None:
                    log.warning("Missing parameter/option: ***" + requiredParam +"*** in file: ***"+fileName+"***")
                else:
                    log.warning("Missing parameter/option: " + requiredParam )
        if not valid:
            log.error("Missing one or parameters specified above. Exiting")
            raise ValueError("Missing one or parameters specified in logging. Exiting")
