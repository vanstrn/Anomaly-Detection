"""
Primary layer function that handles identifying which layers to use when building networks.
"""

import tensorflow as tf
import tensorflow.keras.layers as KL
import json
import .customLayers import *
import tensorflow.keras.backend as K


def GetLayer(layerDict):
    """Based on a layerDictionary input the function returns the appropriate layer for the NN."""

    if layerDict["layerType"] == "Dense":
        layer = KL.Dense( **layerDict["Parameters"],name=layerDict["layerName"])
    elif layerDict["layerType"] == "Conv2D":
        layer = KL.Conv2D( **layerDict["Parameters"],name=layerDict["layerName"])
    elif layerDict["layerType"] == "Conv2DTranspose":
        layer = KL.Conv2DTranspose( **layerDict["Parameters"],name=layerDict["layerName"])
    elif layerDict["layerType"] == "SeparableConv":
        layer = KL.SeparableConv2D( **layerDict["Parameters"],name=layerDict["layerName"])
    elif layerDict["layerType"] == "Flatten":
        layer= KL.Flatten()
    elif layerDict["layerType"] == "AveragePool":
        layer= KL.AveragePooling2D(**layerDict["Parameters"],name=layerDict["layerName"])
    elif layerDict["layerType"] == "GlobalAveragePooling2D":
        layer= KL.GlobalAveragePooling2D(name=layerDict["layerName"])
    elif layerDict["layerType"] == "SoftMax":
        layer = KL.Activation('softmax')
    elif layerDict["layerType"] == "Concatenate":
        layer = KL.Concatenate( **layerDict["Parameters"],name=layerDict["layerName"])
    elif layerDict["layerType"] == "Multiply":
        layer = KL.Multiply( **layerDict["Parameters"],name=layerDict["layerName"])
    elif layerDict["layerType"] == "Add":
        layer = KL.Add( **layerDict["Parameters"],name=layerDict["layerName"])
    elif layerDict["layerType"] == "Reshape":
        layer = KL.Reshape( **layerDict["Parameters"],name=layerDict["layerName"])
    elif layerDict["layerType"] == "LSTM":
        layer = KL.LSTM(**layerDict["Parameters"],name=layerDict["layerName"])
    elif layerDict["layerType"] == "SimpleRNN":
        layer = KL.SimpleRNN(**layerDict["Parameters"],name=layerDict["layerName"])
    elif layerDict["layerType"] == "UpSampling2D":
        layer = KL.UpSampling2D(**layerDict["Parameters"],name=layerDict["layerName"])
    elif layerDict["layerType"] == "GaussianNoise":
        layer = KL.GaussianNoise(**layerDict["Parameters"],name=layerDict["layerName"])
    elif layerDict["layerType"] == "Dropout":
        layer = KL.Dropout(**layerDict["Parameters"],name=layerDict["layerName"])
    elif layerDict["layerType"] == "ZeroPadding2D":
        layer = KL.ZeroPadding2D(**layerDict["Parameters"],name=layerDict["layerName"])

    #Weird Math Layers
    elif layerDict["layerType"] == "LogSoftMax":
        layer = tf.nn.log_softmax
    elif layerDict["layerType"] == "Clip":
        layer = KL.Lambda(lambda x: tf.clip_by_value(x,**layerDict["Parameters"]))
    elif layerDict["layerType"] == "Log":
        layer = tf.math.log
    elif layerDict["layerType"] == "Sum":
        layer = tf.keras.backend.sum
    elif layerDict["layerType"] == "StopGradient":
        layer = KL.Lambda(lambda x: K.stop_gradient(x))
    elif layerDict["layerType"] == "StopNan":
        layer = KL.Lambda(lambda x: tf.math.maximum(x,1E-9))

    #Custom Layers
    elif layerDict["layerType"] == "LSTM_Reshape":
        layer = LSTM_Reshape(**layerDict["Parameters"],name=layerDict["layerName"])
    elif layerDict["layerType"] == "Round":
        layer= RoundingSine(name=layerDict["layerName"])
    elif layerDict["layerType"] == "LSTM_Unshape":
        layer = LSTM_Unshape(**layerDict["Parameters"],name=layerDict["layerName"])
    elif layerDict["layerType"] == "Inception":
        layer = Inception(**layerDict["Parameters"],name=layerDict["layerName"])
    elif layerDict["layerType"] == "ReverseInception":
        layer = ReverseInception(**layerDict["Parameters"],name=layerDict["layerName"])
    elif layerDict["layerType"] == "NonLocalNN":
        layer= Non_local_nn( **layerDict["Parameters"],name=layerDict["layerName"])
    elif layerDict["layerType"] == "Split":
        layer= Split( **layerDict["Parameters"],name=layerDict["layerName"])
    elif layerDict["layerType"] == "BasicCNNSplit":
        layer= BasicCNNSplit( **layerDict["Parameters"],name=layerDict["layerName"])
    elif layerDict["layerType"] == "ChannelFilter":
        layer= ChannelFilter( **layerDict["Parameters"],name=layerDict["layerName"])

    return layer
