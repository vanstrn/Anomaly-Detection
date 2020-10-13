from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import numpy as np

class RoundingSawtooth(Layer):
    """Does approximate rounding with Sawtooth wave."""
    def __init__(self, filters, **kwargs):
        super(RoundingSawtooth, self).__init__(**kwargs)
        self.filters =filters
    def call(self, inputs):
        if isinstance(self.filters,int):
            return
        elif isinstance(self.filters,list):
            return 

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {}
        base_config = super(RoundingSawtooth, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == "__main__":
    from tensorflow.keras.models import Sequential
    import tensorflow as tf
    model=Sequential()
    model.add(RoundingSawtooth())
    x=tf.convert_to_tensor(np.random.random([3,3]), dtype=tf.float32)
    print(x)
    print(model(x))
