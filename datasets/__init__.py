from .mnist import *
from .fmnist import *
from .cifar import *

def LoadDataset(settings,**kwargs):
    func = eval(settings["Dataset"]["Name"])
    return func(**settings["Dataset"]["Arguments"])
