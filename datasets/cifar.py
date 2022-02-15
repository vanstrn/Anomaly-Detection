
import tensorflow as tf
import numpy as np
from utils.math import *

class CIFAR_Basic():
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        lab_train = np.eye(10)[y_train]
        lab_test = np.eye(10)[y_test]
        self.trainData = {
            "image":RGBtoNORM(x_train).astype(np.float32),
            "label":lab_train.astype(np.float32),
        }
        self.testData = {
            "image":RGBtoNORM(x_test).astype(np.float32),
            "label":lab_test.astype(np.int32),
        }

        self.inputSpec = {"image":[32,32,3]}
        self.outputSpec = {"classes":10}


class CIFAR_Anomaly():
    def __init__(self,holdout=0):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        #Removing holdout from training data
        x_train = np.delete(x_train, np.where(y_train == holdout), axis=0)
        #Relabelling
        lab_test = np.zeros_like(y_test)
        lab_test[np.where(y_test==holdout)] = 1.

        self.trainData = {
            "image":RGBtoNORM(x_train).astype(np.float32),
        }
        self.testData = {
            "image":RGBtoNORM(x_test).astype(np.float32),
            "anom_label":lab_test.astype(np.float32),
            "label":y_test.astype(np.int32),
        }

        self.inputSpec = {"image":[32,32,3]}
        self.outputSpec = {"image_size":[32,32,3]}


class CIFAR_RECON():
    def __init__(self):
        (x_train, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        self.trainData = {
            "image":RGBtoNORM(x_train).astype(np.float32)
        }
        self.testData = {
            "image":RGBtoNORM(x_test).astype(np.float32),
            "label":y_test.astype(np.float32),
        }

        self.inputSpec = {"image":[32,32,3]}
        self.outputSpec = {"image_size":[32,32,3]}