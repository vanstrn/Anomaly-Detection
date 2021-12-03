"""
Definition of several classes of schedulers
"""

import numpy as np
import tensorflow as tf

def GetScheduler(schedulerType,**kwargs):
    if schedulerType == "LinearFlat":
        sched = LinearScheduler(**kwargs)
    elif schedulerType == "Step":
        sched = StepScheduler(**kwargs)
    return tf.keras.callbacks.LearningRateScheduler(sched.StepValue)


class LinearScheduler():
    def __init__(self,startValue,endValue,episodeDecay,dtype="float"):
        self.dtype = dtype

        self.startValue = startValue
        self.endValue = endValue
        self.episodeDecay = episodeDecay

    def StepValue(self,epoch, lr):
        if episode<self.episodeDecay:
            value = self.endValue + (self.startValue-self.endValue)*(1-episode/self.episodeDecay)
        else:
            value=self.endvalue

        if self.dtype=="int":
            return int(value)
        elif self.dtype=="float":
            return float(value)


class StepScheduler():
    def __init__(self,startValue,endValue,episodeStep,dtype="float"):
        self.dtype = dtype

        self.startValue = startValue
        self.endValue = endValue
        self.episodeStep = episodeStep

    def StepValue(self,epoch, lr):
        if episode<self.episodeStep:
            value = self.startValue
        else:
            value = self.endValue

        if self.dtype=="int":
            return int(value)
        elif self.dtype=="float":
            return float(value)
