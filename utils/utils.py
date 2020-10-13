# Module contains any methods, class, parameters, etc that is related to logging the trainig

import numpy as np
import os
from importlib import import_module #Used to import module based on a string.
import inspect
import functools
import collections.abc

def LoadConfig(fileName,**kwargs):
    pass

def InitializeBackend(**kwargs):
    print(backend)

def LoadMethod(settingDict,**kwargs):
    # Method = GetFunction(settings["Method"])
    # net = Method(settings)
    print(backend)
    return None


def LoadDataset(settingDict):
    return None
    datasetConfigs = settingDict["Dataset"]

    if datasetConfig["Type"] == "local":
        #Reading in images or data from local disk:
        pass
    elif datasetConfig["Type"] == "web":
        #Downloading Images from websource
        pass
    elif datasetConfig["Type"] == "tensorflow":
        pass
    elif datasetConfig["Type"] == "pytorch":
        pass
    else:
        pass

def UpdateNestedDictionary(defaultSettings,overrides):
    for label,override in overrides.items():
        if isinstance(override, collections.abc.Mapping):
            UpdateNestedDictionary(defaultSettings[label],override)
        else:
            defaultSettings[label] = override
    return defaultSettings


def GetFunction(string):
    module_name, func_name = string.rsplit('.',1)
    module = import_module(module_name)
    func = getattr(module,func_name)
    return func

def CreatePath(path, override=False):
    """
    Create directory
    If override is true, remove the directory first
    """
    if override:
        if os.path.exists(path):
            shutil.rmtree(path,ignore_errors=True)
            if os.path.exists(path):
                raise OSError("Failed to remove path {}.".format(path))

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            raise OSError("Creation of the directory {} failed".format(path))

class MovingAverage:
    """MovingAverage
    Container that only store give size of element, and store moving average.
    Queue structure of container.
    """

    def __init__(self, size):
        """__init__

        :param size: number of element that will be stored in he container
        """
        from collections import deque
        self.average = 0.0
        self.size = size

        self.queue = deque(maxlen=size)

    def __call__(self):
        """__call__"""
        return self.average
    def std(self):
        """__call__"""
        if len(self.queue) < 2:
            return 1
        return np.std(np.asarray(self.queue))

    def tolist(self):
        """tolist
        Return the elements in the container in (list) structure
        """
        return list(self.queue)

    def extend(self, l: list):
        """extend

        Similar to list.extend

        :param l (list): list of number that will be extended in the deque
        """
        # Append list of numbers
        self.queue.extend(l)
        self.size = len(self.queue)
        self.average = sum(self.queue) / self.size

    def append(self, n):
        """append

        Element-wise appending in the container

        :param n: number that will be appended on the container.
        """
        s = len(self.queue)
        if s == self.size:
            self.average = ((self.average * self.size) - self.queue[0] + n) / self.size
        else:
            self.average = (self.average * s + n) / (s + 1)
        self.queue.append(n)

    def clear(self):
        """clear
        reset the container
        """
        self.average = 0.0
        self.queue.clear()
