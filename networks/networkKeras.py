"""
Sets up the basic Network Class which lays out all required functions of a Neural Network.

"""
import logging
logger = logging.getLogger(__name__)

import json
from tensorflow.keras import backend as K
from tensorflow import keras

from pprint import pformat
import os

from .layers import GetLayer
from .utils import *



def CreateModel(networkConfigFile, observationSpec, variables={}, scope=None, training=True, printSummary=False):
    """
    Reads a network config file and processes that into a netowrk with appropriate naming structure.

    This class only works on feed forward neural networks. Can only handle one input.

    Parameters
    ----------
    networkConfigFile : str
        Config file which points to the network description to be loaded.

    observationSpec : dict
        Dictionary detailing inputs. Keys are input names, values are input sizes.
    variables : dict
        Dictionary of values which will override the default variables of the config file.
    scope : str [opt]
        Defines a tensorflow scope for the network.

    Returns
    -------
    Keras Functional Model
    """

    networkConfig = LoadJSONFile(networkConfigFile)

    #Creating Recursion sweep to go through dictionaries and lists in the networkConfig to insert user defined values.
    if "DefaultParams" in networkConfig.keys():
        variableFinal = networkConfig["DefaultParams"]
        variableFinal.update(variables)
    else:
        variableFinal = variables
    networkConfig["NetworkStructure"] = UpdateStringValues(networkConfig["NetworkStructure"],variableFinal)

    inputs = {}
    outputs = {}
    layers = {}
    interOutputs = {}

    logger.debug("Beginning creation of Network defined by: {}".format(networkConfigFile))
    #Creating All of the inputs
    logger.debug("Defining all inputs for the NN")
    for name_i,input_i in observationSpec.items():
        logger.debug("Building Input: {}".format(name_i))
        tmp = keras.Input(input_i,name=name_i)
        inputs[name_i] = tmp
        interOutputs["input."+name_i] = tmp

    logger.debug("Beginning Creation of network Layers")
    for sectionName,layerList in networkConfig["NetworkStructure"].items():
        for layerDict in layerList:
            logger.debug("Building Layer: {}".format(layerDict["layerName"]))

            if "ReuseLayer" in layerDict:
                layer = layers[layerDict["ReuseLayer"]]
            else:
                layer = GetLayer(layerDict)

            # Finding the inputs to the layer and applying the layer.
            if isinstance(layerDict["layerInput"],list):
                layerInputs = []
                for layerInput in layerDict["layerInput"]:
                    layerInputs.append(interOutputs[layerInput])
                if layerDict["layerType"] in ["Concatenate","Multiply","Add"]:
                    out = layer(layerInputs)
                else:
                    out = layer(*layerInputs)
            ## Modification to apply one network to a set of inputs
            elif "InputDuplicate" in layerDict:
                out = [layer(interOutputs[layerDict["layerInput"]+"_"+str(i)]) for i in range(layerDict["Duplicate"])]

            ## Modification to input a set of inputs to a network.
            elif "InputCombine" in layerDict:
                layerInputs = []
                for i in range(layerDict["InputCombine"]):
                    layerInputs.append(interOutputs[layerDict["layerInput"]+"_"+str(i)])
                out = layer(layerInputs)
            else:
                layerInputs = interOutputs[layerDict["layerInput"]]
                out = layer(layerInputs)

            logger.debug("Built Layer with \n\tLayer Inputs: {}\n\tLayer Outpus: {}".format(layerInputs,out))

            #Way to handle multiple outputs from a layer (Fully and Partially Enumerated.)
            if "MultiOutput" in layerDict:
                #Fully enumerated outputs
                if isinstance(layerDict["MultiOutput"],list):
                    layers[layerDict["layerName"]] = layer
                    for i,name in enumerate(layerDict["MultiOutput"]):
                        interOutputs[name] = out[i]
                        if "Output" in layerDict:
                            if layerDict["Output"]:
                                outputs[name] = out[i]
                #Partially enumerated outputs (Know there are N outputs.)
                elif isinstance(layerDict["MultiOutput"],int):
                    layers[layerDict["layerName"]] = layer
                    for i in range(layerDict["MultiOutput"]):
                        name = "{}_{}".format(layerDict["layerName"],i)
                        interOutputs[name] = out[i]
                        if "Output" in layerDict:
                            if layerDict["Output"]:
                                outputs[name] = out[i]

            else:
                layers[layerDict["layerName"]] = layer
                interOutputs[layerDict["layerName"]] = out
                if "Output" in layerDict:
                    if layerDict["Output"]:
                        outputs[layerDict["layerName"]] = out

            #Adding the appropriate layers to the output.

    model = keras.Model(inputs,outputs,name=scope)

    if printSummary:
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        logger.info("\n\t".join(stringlist))


    return model

if __name__ == "__main__":
    pass
