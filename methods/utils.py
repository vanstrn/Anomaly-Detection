import tensorflow as tf

from utils.utils import GetFunction
import logging
import numpy as np

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

@tf.function
def VanishingGradient(gradients,limit=1e-8):
    """Takes a list of computed gradients and analyzes the ratio of gradients below a threshold,
    producing a metric measuring the changes during learning. """
    totalCounter = 1
    vanishCounter = 0
    for gradient in gradients:
        if gradient is not None:
            vanished=tf.math.less_equal(tf.math.abs(gradient),tf.constant(limit,dtype=tf.float32))
            #Measuring if they are idetically zero. These are excluded from the count.
            non_zero=tf.math.greater(tf.math.abs(gradient),tf.constant(0.0,dtype=tf.float32))
            both=tf.math.logical_and(vanished,non_zero)

            totalCounter += tf.reduce_sum(tf.cast(non_zero,tf.int32))
            vanishCounter += tf.reduce_sum(tf.cast(both,tf.int32))
    return vanishCounter/totalCounter
