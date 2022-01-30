
import argparse
import tensorflow as tf
import os
import json
import numpy as np
import logging
from utils.utils import LoadConfig,CreatePath,GetFunction,UpdateNestedDictionary
from utils.utils import LoadDataset,Logger,JSON_Load
from methods.utils import LoadMethod

log = logging.getLogger(__name__)

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

#Setting up tensorflow based on specified processor configur
if "gpu" in args.processor:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.keras.backend.set_floatx('float32')

#Creating specific save folders for the different
logger = Logger('./logs/'+settings["RunName"])
logger.RecordGitState()

log.info("Running the expeiment with {}. CUDA_VISIBLE_DEVICES={}".format(args.processor,os.getenv("CUDA_VISIBLE_DEVICES")))

if args.seed is None:
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
            callbacks.append(func(logger,dataset,**dict["Arguments"]))

    if args.test:
        #Skipping Training and going straight to testing. This calls "On Train End" Callbacks
        model.Test(callbacks=callbacks)
    else:
        model.Train(dataset.trainData,callbacks=callbacks)


except Exception as e:
    log.warning("Closing environment due to error")
    log.error('Error', exc_info=e)
except KeyboardInterrupt:
    log.warning("Closing environment due to Keyboard Interrupt")
