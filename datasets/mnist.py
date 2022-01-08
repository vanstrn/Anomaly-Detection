"""
This file sets up interesting subsets of data
"""

import tensorflow as tf
import numpy as np
from .utils import *

class MNIST_Basic():
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
        lab_train = np.eye(10)[y_train]
        lab_test = np.eye(10)[y_test]
        self.trainData = {
            "image":NormalizeImages(x_train).astype(np.float32),
            "label":lab_train.astype(np.float32),
        }
        self.testData = {
            "image":NormalizeImages(x_test).astype(np.float32),
            "label":lab_test.astype(np.float32),
        }

        self.inputSpec = {"image":[28,28,1]}
        self.outputSpec = {"classes":10}


class MNIST_Anomaly():
    def __init__(self,holdout=0):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
        #Removing holdout from training data
        x_train = np.delete(x_train, np.where(y_train == holdout), axis=0)
        #Relabelling
        lab_test = np.zeros_like(y_test)
        lab_test[np.where(y_test==holdout)] = 1.

        self.trainData = {
            "image":NormalizeImages(x_train).astype(np.float32),
        }
        self.testData = {
            "image":NormalizeImages(x_test).astype(np.float32),
            "anom_label":lab_test.astype(np.float32),
        }

        self.inputSpec = {"image":[28,28,1]}
        self.outputSpec = {"image_size":[28,28]}


class MNIST_RECON():
    def __init__(self):
        (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data(path='mnist.npz')



        self.trainData = {
            "image":NormalizeImages(x_train).astype(np.float32)
        }
        self.testData = {
            "image":NormalizeImages(x_test).astype(np.float32)
        }

        self.inputSpec = {"image":[28,28,1]}
        self.outputSpec = {"image_size":[28,28]}
