#!/usr/bin/env python
# Created By  : Neale Van Stralen
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Methods for loading MNIST Digits dataset for classification, reconstruction, and anomaly detection.
"""
# ---------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
from utils.math import *
from sklearn.model_selection import train_test_split

class MNIST_Basic():
    """Basic MNIST Digits dataset setup for classification.
    Data is split into 3 partitions: training, validation, and testing.
    Images are labeled as a one-hot class label for compatibility with neural network vector outputs."""

    def __init__(self,validationSplit=0.10):

        (x_learn, y_learn), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
        x_train, x_valid, y_train, y_valid = train_test_split(x_learn, y_learn, train_size=(1-validationSplit),
                                                  random_state=42, shuffle=True)
        lab_train = np.eye(10)[y_train]
        lab_valid = np.eye(10)[y_valid]
        lab_test = np.eye(10)[y_test]

        self.trainData = {
            "image":np.expand_dims(RGBtoNORM(x_train).astype(np.float32),axis=-1),
            "label":lab_train.astype(np.float32),
        }
        self.validationData = {
            "image":np.expand_dims(RGBtoNORM(x_valid).astype(np.float32),axis=-1),
            "label":lab_valid.astype(np.int32),
        }
        self.testData = {
            "image":np.expand_dims(RGBtoNORM(x_test).astype(np.float32),axis=-1),
            "label":lab_test.astype(np.int32),
        }

        self.inputSpec = {"image":[28,28,1]}
        self.outputSpec = {"classes":10}


class MNIST_RECON():
    """Basic MNIST Digits dataset setup for reconstruction.
    Data is split into 3 partitions: training, validation, and testing.
    Training data is unlabeled. Validation and test images have class labels for processing."""

    def __init__(self,validationSplit=0.10):

        (x_learn, y_learn), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
        x_train, x_valid, y_train, y_valid = train_test_split(x_learn, y_learn, train_size=(1-validationSplit),
                                                  random_state=42, shuffle=True)

        self.trainData = {
            "image":np.expand_dims(RGBtoNORM(x_train).astype(np.float32),axis=-1),
        }
        self.validationData = {
            "image":np.expand_dims(RGBtoNORM(x_valid).astype(np.float32),axis=-1),
            "label":y_valid.astype(np.int32),
        }
        self.testData = {
            "image":np.expand_dims(RGBtoNORM(x_test).astype(np.float32),axis=-1),
            "label":y_test.astype(np.float32),
        }

        self.inputSpec = {"image":[28,28,1]}
        self.outputSpec = {"image_size":[28,28,1]}


class MNIST_Anomaly():
    """Basic MNIST Digits dataset setup for anomaly detection.
    Anomalies are created by withholding one of the ten classes during training amd labelling it as the anomaly.
    Data is split into 3 partitions: training, validation, and testing.
    Training data is unlabeled. Validation and test images have class labels and anomaly markers for processing."""

    def __init__(self,holdout=0,validationSplit=0.10):
        (x_learn, y_learn), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
        x_train, x_valid, y_train, y_valid = train_test_split(x_learn, y_learn, train_size=(1-validationSplit),
                                                  random_state=42, shuffle=True)
        #Removing holdout from training data
        x_train = np.delete(x_train, np.where(y_train == holdout), axis=0)

        #Creating label for anomaly detection
        lab_valid = np.zeros_like(y_valid)
        lab_valid[np.where(y_valid==holdout)] = 1.
        lab_test = np.zeros_like(y_test)
        lab_test[np.where(y_test==holdout)] = 1.

        self.trainData = {
            "image":np.expand_dims(RGBtoNORM(x_train).astype(np.float32),axis=-1),
        }
        self.validationData = {
            "image":np.expand_dims(RGBtoNORM(x_valid).astype(np.float32),axis=-1),
            "anom_label":lab_valid.astype(np.float32),
            "label":y_valid.astype(np.int32),
        }
        self.testData = {
            "image":np.expand_dims(RGBtoNORM(x_test).astype(np.float32),axis=-1),
            "anom_label":lab_test.astype(np.float32),
            "label":y_test.astype(np.int32),
        }

        self.inputSpec = {"image":[28,28,1]}
        self.outputSpec = {"image_size":[28,28,1]}


class MNIST_OC_Anomaly():
    """Basic MNIST Digits dataset setup for anomaly detection.
    Anomalies are created by training on only one of the ten classes during training amd labelling the rest as anomalies.
    Data is split into 3 partitions: training, validation, and testing.
    Training data is unlabeled. Validation and test images have class labels and anomaly markers for processing."""

    def __init__(self,holdout=0,validationSplit=0.10):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
        x_train, x_valid, y_train, y_valid = train_test_split(x_learn, y_learn, train_size=(1-validationSplit),
                                                  random_state=42, shuffle=True)
        #Removing holdout from training data
        x_train = np.delete(x_train, np.where(y_train != holdout), axis=0)

        #Creating label for anomaly detection
        lab_valid = np.zeros_like(y_valid)
        lab_valid[np.where(y_valid!=holdout)] = 1.
        lab_test = np.zeros_like(y_test)
        lab_test[np.where(y_test!=holdout)] = 1.

        self.trainData = {
            "image":np.expand_dims(RGBtoNORM(x_train).astype(np.float32),axis=-1),
        }
        self.validationData = {
            "image":np.expand_dims(RGBtoNORM(x_valid).astype(np.float32),axis=-1),
            "anom_label":lab_valid.astype(np.float32),
            "label":y_valid.astype(np.int32),
        }
        self.testData = {
            "image":np.expand_dims(RGBtoNORM(x_test).astype(np.float32),axis=-1),
            "anom_label":lab_test.astype(np.float32),
            "label":y_test.astype(np.int32),
        }

        self.inputSpec = {"image":[28,28,1]}
        self.outputSpec = {"image_size":[28,28,1]}
