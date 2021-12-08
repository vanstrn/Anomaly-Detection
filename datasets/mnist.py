"""
This file sets up interesting subsets of data
"""

import tensorflow as tf
import numpy as np


class MNIST_Basic():
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        lab_train = np.eye(10)[y_train]
        lab_test = np.eye(10)[y_test]
        self.trainData = {
            "x_train":x_train,
            "y_train":lab_train,
        }
        self.testData = {
            "x_test":x_test,
            "y_test":lab_test,
        }

        self.inputSpec = {"image":[28,28,1]}
        self.outputSpec = {"classes":10}


class MNIST_Anomaly():
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        lab_train = np.eye(10)[y_train]
        lab_test = np.eye(10)[y_test]
        self.trainData = {
            "x_train":x_train,
            "y_train":lab_train,
        }
        self.testData = {
            "x_test":x_test,
            "y_test":lab_test,
        }

        self.inputSpec = {"image":[28,28,1]}
        self.outputSpec = {"classes":10}
