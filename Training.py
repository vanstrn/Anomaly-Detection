#!/usr/bin/env python
# Created By  : Neale Van Stralen
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Main executable for training of machine vision experiments.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import argparse
from utils.utils import JSON_Load,LoadConfig,GetFunction,UpdateNestedDictionary
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--configFile", required=True,
    help="Filename or filepath to be run.")
parser.add_argument("-c", "--config", required=False, default={}, action=JSON_Load,
    help="Specify overrides to be made to the run.")
parser.add_argument("-n", "--network", required=False, default={}, action=JSON_Load,
    help="Specify special parameters for the network.")
parser.add_argument("-p", "--processor", required=False, default="gpu0",
    help="Processor identifier string. [cpu / gpu0 / gpu1]")
parser.add_argument("--test", required=False,
    action="store_true",help="Flag to skip testing")
parser.add_argument("--seed", required=False,
    default=None, help="Seed input to set tensorflow and numpy seed.")
args = parser.parse_args()
#Reading in the config files.
settings = LoadConfig(args.configFile)
settings = UpdateNestedDictionary(settings,args.config)

LOG_PATH = './logs/'+settings["RunName"]

logging.basicConfig(
    level=logging.INFO,
    # format='%(levelname)s -- %(asctime)s -- %(message)s',
    format='%(levelname)s - %(asctime)s - (%(filename)s:%(lineno)d) - %(message)s ',
    # format=CustomFormatter(),
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH + '/logger.log'),
        ],
    datefmt='%Y/%m/%d %I:%M:%S %p'
    )
logger = logging.getLogger("Trainer")

logger.info("Running experiment with the following config properties:\n{}".format(pformat(settings)))

import tensorflow as tf
import numpy as np
import random
from pprint import pformat

from utils.logging import ExperimentLogger
from methods.utils import LoadMethod
from datasets import LoadDataset

#Setting up tensorflow based on specified processor configuration
if "gpu" in args.processor:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.keras.backend.set_floatx('float32')
logger.info("Running the expeiment with {}. CUDA_VISIBLE_DEVICES={} \n\t Actual GPUs: {}".format(args.processor,os.getenv("CUDA_VISIBLE_DEVICES"),gpus))

#Creating experiment logger which with functions for saving tensorboard logs and images
expLogger = ExperimentLogger(LOG_PATH)
expLogger.RecordGitState()

if args.seed is not None:
    seed = args.seed
else:
    seed = random.randint(0, 1000000)
logger.info("Running experiment using seed: `{}`".format(seed))
tf.random.set_seed(args.seed)
np.random.seed(args.seed)

try:
    dataset = LoadDataset(settings)

    model = LoadMethod(settings,dataset,networkConfig=args.network)
    # Initialize Callbacks
    callbacks=[]
    if "Callbacks" in settings:
        for dict in settings["Callbacks"]:
            func = GetFunction(dict["Name"])
            callbacks.append(func(expLogger,dataset,**dict["Arguments"]))

    if args.test:
        #Skipping Training and going straight to testing. This calls "On Train End" Callbacks
        model.Test(callbacks=callbacks)
    else:
        model.Train(dataset.trainData,callbacks=callbacks)

except Exception as e:
    logger.warning("Closing environment due to error")
    logger.error('Error', exc_info=e)
except KeyboardInterrupt:
    logger.warning("Closing environment due to User Interrupt")
