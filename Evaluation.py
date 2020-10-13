
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--backend", required=False, default="Tensorflow",
                    help="Specify which NN backend to use. Options [Tensorflow / PyTorch]")
parser.add_argument("-f", "--configFile", required=True,
                    help="Filename or filepath to be run.")
parser.add_argument("-o", "--override", required=False, default={},
                    help="Specify overrides to be made to the run.")
parser.add_argument("-p", "--processor", required=False, default="gpu0",
                    help="Processor identifier string. [cpu / gpu0 / gpu1]")

args = parser.parse_args()

#Reading in the config files.
settings = LoadConfig(args.configFile)

#Loading the Model
model = LoadMethod(settings)

#Loading the DataSet
dataset = LoadDataset(settings)

#Training
model.evaluate(dataset)
