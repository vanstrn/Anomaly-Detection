import tensorflow as tf

from utils.utils import GetFunction

def LoadMethod(settingDict,**kwargs):
    Method = GetFunction(settingDict["Method"])
    net = Method(settingDict)
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
