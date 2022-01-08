# Module contains any methods, class, parameters, etc that is related to logging the trainig

import numpy as np
import os
from importlib import import_module #Used to import module based on a string.
from urllib.parse import unquote
import collections.abc
import json
from datasets import *
import logging
import tensorflow as tf
from utils.git import SaveCurrentGitState
import argparse
import matplotlib.pyplot as plt

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


def LoadDataset(settings,**kwargs):
    func = eval(settings["Dataset"]["Name"])
    return func(**settings["Dataset"]["Arguments"])


class Logger():
    def __init__(self, LOG_PATH):
        self.LOG_PATH = LOG_PATH

        CreatePath(LOG_PATH)
        CreatePath(LOG_PATH+'/images')
        CreatePath(LOG_PATH+'/tb_logs')

        self.writer = tf.summary.create_file_writer(LOG_PATH+'/tb_logs')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(LOG_PATH + '/logger.log'),
                ],
            datefmt='%Y/%m/%d %I:%M:%S %p'
            )

    def LogScalar(self, tag,value,step):
        with self.writer.as_default():
            summary = tf.summary.scalar(tag,value,step=tf.cast(step, tf.int64))
        self.writer.flush()

    def SaveImage(self,name,format="png"):
        plt.savefig("{}/images/{}.{}".format(self.LOG_PATH,name,format))
        plt.close()

    def LogImage(self,image,name,step):
        with self.writer.as_default():
            summary = tf.summary.image(name, image, step=step)
        self.writer.flush()

    def LogMatplotLib(self,figure,name,step):
        with self.writer.as_default():
            summary = tf.summary.image(name, plot_to_image(figure), step=step)
        self.writer.flush()

    def RecordGitState(self):
        SaveCurrentGitState(self.LOG_PATH)

import io

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image
