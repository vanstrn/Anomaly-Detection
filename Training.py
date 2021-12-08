
import argparse
import tensorflow as tf
import json
from urllib.parse import unquote

from utils.utils import LoadConfig,CreatePath,GetFunction,UpdateNestedDictionary
from datasets.utils import LoadDataset
from methods.utils import LoadMethod
from utils.git import SaveCurrentGitState

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--backend", required=False, default="Tensorflow1",
                    help="Specify which NN backend to use.",
                    choices=["Tensorflow1" , "Tensorflow2" , "PyTorch"])
parser.add_argument("-f", "--configFile", required=True,
                    help="Filename or filepath to be run.")
parser.add_argument("-c", "--config", required=False, default=None,
                    help="Specify overrides to be made to the run.")
parser.add_argument("-n", "--network", required=False, default=None,
                    help="Specify special parameters for the network.")
parser.add_argument("-p", "--processor", required=False, default="gpu0",
                    help="Processor identifier string. [cpu / gpu0 / gpu1]")

args = parser.parse_args()
if args.config is not None: configOverride = json.loads(unquote(args.config))
else: configOverride = {}
if args.network is not None: networkVariables = json.loads(unquote(args.network))
else: networkVariables = {}

#Reading in the config files.
settings = LoadConfig(args.configFile)
settings = UpdateNestedDictionary(settings,configOverride)

#Setting up tensorflow based on specified processor configur
if "gpu" in args.processor:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# logger.info("Running the expeiment with {}. CUDA_VISIBLE_DEVICES={}".format(args.processor,os.getenv("CUDA_VISIBLE_DEVICES")))
tf.keras.backend.set_floatx('float32')

#Creating specific save folders for the different
EXP_NAME = settings["RunName"]
MODEL_PATH = './models/'+EXP_NAME
LOG_PATH = './logs/'+EXP_NAME
CreatePath(LOG_PATH)
CreatePath(MODEL_PATH)

#Saving Current Git Status for replication purposes.
SaveCurrentGitState(LOG_PATH)

########## Loading the DataSet ########
dataset = LoadDataset(settings)

print(dataset)
#Loading the Model
model = LoadMethod(settings,dataset,networkConfig=networkVariables)

# Initialize Callbacks
callbacks=[]
if "Callbacks" in settings:
    for dict in settings["Callbacks"]:
        func = GetFunction(dict["Name"])
        callbacks.append(func(LOG_PATH,dataset,**dict["Arguments"]))

print(callbacks)
########## Training ############################################################
model.Train(dataset.trainData,callbacks=callbacks)

model.Test(dataset.testData)
