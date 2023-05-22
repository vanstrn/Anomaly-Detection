"""
Definition of several classes of schedulers
"""

import numpy as np
import tensorflow as tf

def GetScheduler(schedulerType,**kwargs):
    if schedulerType == "Constant":
        return ConstantScheduler(**kwargs)
    elif schedulerType == "LinearLinear":
        return LinearScheduler(**kwargs)
    elif schedulerType == "OffsetLinear":
        return OffsetLinearScheduler(**kwargs)
    elif schedulerType == "Step":
        return StepScheduler(**kwargs)
    elif schedulerType == "Square":
        return SquareWaveScheduler(**kwargs)
    return tf.keras.callbacks.LearningRateScheduler(sched.StepValue)


class ConstantScheduler():
    def __init__(self,value,dtype="float"):
        self.dtype = dtype

        if self.dtype=="int":
            self.value = int(value)
        elif self.dtype=="float":
            self.value = float(value)

    def StepValue(self,**kwargs):
        return self.value


class LinearScheduler():
    def __init__(self,startValue,endValue,episodeDecay,dtype="float"):
        self.dtype = dtype

        self.startValue = startValue
        self.endValue = endValue
        self.episodeDecay = episodeDecay

    def StepValue(self,episode,**kwargs):
        if episode<self.episodeDecay:
            value = self.endValue + (self.startValue-self.endValue)*(1-episode/self.episodeDecay)
        else:
            value=self.endvalue

        if self.dtype=="int":
            return int(value)
        elif self.dtype=="float":
            return float(value)


class OffsetLinearScheduler():
    def __init__(self,startValue,endValue,startEpisode,episodeDecay,dtype="float"):
        self.dtype = dtype

        self.startValue = startValue
        self.endValue = endValue
        self.startEpisode = startEpisode
        self.episodeDecay = episodeDecay

    def StepValue(self,episode,**kwargs):
        if episode<self.startEpisode:
            value = self.startValue
        elif episode<(self.episodeDecay+self.startEpisode):
            episode_ = episode-self.startEpisode
            value = self.endValue + (self.startValue-self.endValue)*(1-episode_/self.episodeDecay)
        else:
            value=self.endValue

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

    def StepValue(self,episode,**kwargs):
        if episode<self.episodeStep:
            value = self.startValue
        else:
            value = self.endValue

        if self.dtype=="int":
            return int(value)
        elif self.dtype=="float":
            return float(value)


class SquareWaveScheduler():
    def __init__(self,highValue,lowValue,highDuration,lowDuration,offset=0,startHigh=True,dtype="float"):
        self.dtype = dtype

        self.highValue = highValue
        self.lowValue = lowValue
        self.highDuration = highDuration
        self.lowDuration = lowDuration
        self.offset = offset
        self.startHigh = startHigh

        if self.startHigh:
            self.nextSwitch = self.highDuration + self.offset
            self.value = self.highValue
            self.high = True
        else:
            self.nextSwitch = self.lowDuration + self.offset
            self.value = self.lowValue
            self.high = False


    def StepValue(self,episode,**kwargs):
        if episode>self.nextSwitch:
            if self.high:
                self.nextSwitch = self.lowDuration + episode
                self.value = self.lowValue
                self.high=False
            else:
                self.nextSwitch = self.highDuration + episode
                self.value = self.highValue
                self.high=True

        if self.dtype=="int":
            return int(self.value)
        elif self.dtype=="float":
            return float(self.value)
        elif self.dtype=="bool":
            return bool(self.value)
