#!/usr/bin/env python
# Created By  : Neale Van Stralen
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Module contains generic methods, classes, parameters, etc.
"""
# ---------------------------------------------------------------------------

import numpy as np
import os
from importlib import import_module #Used to import module based on a string.
from urllib.parse import unquote
import collections.abc
import json
import argparse

class JSON_Load(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super(JSON_Load, self).__init__(option_strings,dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        jsonDict = json.loads(unquote(values))
        setattr(namespace, self.dest, jsonDict)


def LoadConfig(targetFileName,**kwargs):
    for (dirpath, dirnames, filenames) in os.walk("runConfigs"):
        for filename in filenames:
            if targetFileName == filename:
                runConfigFile = os.path.join(dirpath,filename)
                break
    with open(runConfigFile) as json_file:
        settings = json.load(json_file)
    return settings


def CheckFilled(requiredParams,dictionary,fileName=None):
    valid = True
    for requiredParam in requiredParams:
        if requiredParam not in dictionary:
            valid = False
            if fileName is not None:
                print("Missing parameter/option: ***" + requiredParam +"*** in file: ***"+fileName+"***")
            else:
                print("Missing parameter/option: " + requiredParam )
    if not valid:
        print("Missing one or parameters. Exiting")
        exit()


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


def MergeDictValues(listInputDict):
    """Calculating average values of keys in dict"""
    outputDict={}
    for k in listInputDict[0].keys():
        outputDict[k] = np.mean([d[k] for d in listInputDict])
    return outputDict


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
