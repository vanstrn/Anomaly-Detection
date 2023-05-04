#!/usr/bin/env python
# Created By  : Neale Van Stralen
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Utility functions for reading and processing Tensorflow Logs.

"""
# ---------------------------------------------------------------------------
import tensorflow as tf
from scipy.signal import savgol_filter
import numpy as np
import os
import re

from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.python.framework import tensor_util

def GetFileContents(filename):
    """Reads a tensorflow summary file and extracts data into a structured dictionary."""
    dataDict = {}
    for e in summary_iterator(filename):
        for v in e.summary.value:
            if v.tag not in dataDict:
                # print(v.tag)
                dataDict[v.tag] = {}
                dataDict[v.tag]["data"] = []
                dataDict[v.tag]["step"] = []
            # t = tensor_util.MakeNdarray(v.simple_value)
            if v.simple_value != 0.0:
                dataDict[v.tag]["data"].append(v.simple_value)
            else:
                try:
                    t = tensor_util.MakeNdarray(v.tensor)
                    dataDict[v.tag]["data"].append(float(t))
                except:
                    dataDict[v.tag]["data"].append(0)

            dataDict[v.tag]["step"].append(e.step)

    return dataDict


def GetValidationStop(validationData,smoothed=False):
    """Identifies the appropriate epoch stop for the experiment based on the results of the validation data."""
    if smoothed:
        validationData = savgol_filter(validationData, 15, 5)

    index = np.argmax(validationData)
    max = validationData[index]
    return max,index


def CollectDataGroupings(dataSeparations,basePath="./logs",superName="",dataLabels=None):
    """Searches through folders to load tensorboard data where the experiment name matches the high level search criteria.
        Useful for collecting sets of replicate experiments for processing

        ***Note that this function has issues if there is more than one tensorboard log in each corresponding folder. ***
        """

    if dataLabels is None:
        dataLabels = dataSeparations

    dataFiles = {}
    for name,label in zip(dataSeparations,dataLabels):
        dataFiles[label]=[]
        for (dirpath, dirnames, filenames) in os.walk(basePath):
            if len(filenames) != 0:
                if superName in dirpath:
                    if bool(re.search(name,dirpath)):
                        for filename in filenames:
                            if "events.out.tfevents" in filename:
                                dataFiles[label].append(GetFileContents(dirpath+"/"+filename))
                                continue
    return dataFiles


def GetAverageAUC(dataSeparations,smoothed=False,**kwargs):
    data = CollectDataGroupings(dataSeparations,**kwargs)
    finalData={}
    for dataName,dataGroup in data.items():
        maxList=[]
        for dataTrial in dataGroup:
            _,i = GetValidationStop(dataTrial["AUC/Validation"]["data"],smoothed)
            maxList.append(dataTrial["AUC/Test"]["data"][i])
        finalData[dataName]=np.average(maxList)
    return finalData


if __name__ == "__main__":
    x = GetAverageAUC(["AE_AD_Holdout0"],smoothed=True)
    print(x)
