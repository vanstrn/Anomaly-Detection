from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import numpy as np
import tensorflow as tf

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def __init__(self,  **kwargs):
        super(Sampling, self).__init__(**kwargs)

    def call(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class SamplingLike(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def __init__(self,  **kwargs):
        super(SamplingLike, self).__init__(**kwargs)

    def call(self, z_mean, z_log_var, phi):
        batch = tf.shape(phi)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
