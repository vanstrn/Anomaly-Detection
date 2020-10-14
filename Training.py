
import argparse
from utils.utils import LoadConfig,LoadMethod,LoadDataset
from utils.backends import InitializeBackend

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--backend", required=False, default="Tensorflow1",
                    help="Specify which NN backend to use.",
                    choices=["Tensorflow1" , "Tensorflow2" , "PyTorch"])
parser.add_argument("-f", "--configFile", required=True,
                    help="Filename or filepath to be run.")
parser.add_argument("-o", "--override", required=False, default={},
                    help="Specify overrides to be made to the run.")
parser.add_argument("-p", "--processor", required=False, default="gpu0",
                    help="Processor identifier string. [cpu / gpu0 / gpu1]")

args = parser.parse_args()
backend = args.backend #Globally defined variable that is used to select backend.

#Reading in the config files.
settings = LoadConfig(args.configFile)

#Loading the Model
# InitializeBackend()
model = LoadMethod(settings,processor=args.processor)

########## Loading the DataSet ########
# dataset = LoadDataset(settings)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# dataset is a dictionary of lists, typically:
# dataset = {"images": [Array of images],
#            "labels": [Array of labels]}

#Training
model.Train((x_train, y_train))
