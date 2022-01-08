
from tensorflow.keras.layers import Layer
import tensorflow.keras.utils
from tensorflow.python.keras.utils import conv_utils

from tensorflow.python.keras.engine.input_spec import InputSpec
class CentralCropping2D(Layer):
    """Cropping layer for 2D input (e.g. picture).
    It crops along spatial dimensions, i.e. height and width.
    Examples:
    >>> input_shape = (2, 28, 28, 3)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> y = tf.keras.layers.CentralCropping2D(cropping=((24, 24))(x)
    >>> print(y.shape)
    (2, 24, 24, 3)
    Args:
    cropping: Int, or tuple of 2 ints,
      - If int: the same symmetric cropping
        is applied to height and width.
      - If tuple of 2 ints:
        interpreted as two different
        symmetric cropping values for height and width:
        `(symmetric_height_crop, symmetric_width_crop)`.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    Input shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
      `(batch_size, rows, cols, channels)`
    - If `data_format` is `"channels_first"`:
      `(batch_size, channels, rows, cols)`
    Output shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
      `(batch_size, cropped_rows, cropped_cols, channels)`
    - If `data_format` is `"channels_first"`:
      `(batch_size, channels, cropped_rows, cropped_cols)`
    """

    def __init__(self, cropping, data_format=None, **kwargs):
        super(CentralCropping2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        if isinstance(cropping, int):
            self.cropping = (cropping, cropping)
        elif hasattr(cropping, '__len__'):
            if len(cropping) != 2:
                raise ValueError('`cropping` should have two elements. '
                             f'Received: {cropping}.')
            self.cropping = conv_utils.normalize_tuple(
                cropping, 2, '1st entry of cropping')
        else:
            raise ValueError('`cropping` should be either an int, '
                           'or a tuple of 2 ints '
                           '(height_crop, width_crop). '
                           f'Received: {cropping}.')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        print(self.input_spec)
        input_shape = tf.TensorShape(input_shape).as_list()
        # pylint: disable=invalid-unary-operand-type
        if self.data_format == 'channels_first':
            return tf.TensorShape([
              input_shape[0], input_shape[1], self.cropping[0], self.cropping[1]
            ])
        else:
            return tf.TensorShape([
              input_shape[0], self.cropping[0], self.cropping[1], input_shape[3]
            ])
        # pylint: enable=invalid-unary-operand-type

    def call(self, inputs):
        # pylint: disable=invalid-unary-operand-type
        if self.data_format == 'channels_first':
            if ((inputs.shape[2] is not None and self.cropping[0] >= inputs.shape[2]) or
                (inputs.shape[3] is not None and self.cropping[1] >= inputs.shape[3])):
                raise ValueError('Argument `cropping` must be '
                             'greater than the input shape. Received: inputs.shape='
                             f'{inputs.shape}, and cropping={self.cropping}')
          #Calculating the centered cropping area:
            if inputs.shape[2] is None:
                # return inputs[:, :, :self.cropping[0], :self.cropping[1]]
                lower0 = (self.input_shape[2]-self.cropping[0])//2
                upper0 = lower0 + self.cropping[0]
                lower1 = (self.input_shape[3]-self.cropping[1])//2
                upper1 = lower1 + self.cropping[1]
            else:
                lower0 = (inputs.shape[2]-self.cropping[0])//2
                upper0 = lower0 + self.cropping[0]
                lower1 = (inputs.shape[3]-self.cropping[1])//2
                upper1 = lower1 + self.cropping[1]
                return inputs[:, :, lower0:upper0, lower1:upper1]
        else:
            if ((inputs.shape[1] is not None and self.cropping[0] >= inputs.shape[1]) or
                (inputs.shape[2] is not None and self.cropping[1] >= inputs.shape[2])):
                raise ValueError('Argument `cropping` must be '
                                 'greater than the input shape. Received: inputs.shape='
                                 f'{inputs.shape}, and cropping={self.cropping}')

            if inputs.shape[2] is None:
                # return inputs[:, :self.cropping[0], :self.cropping[1], :]
                lower0 = (self.input_shape[1]-self.cropping[0])//2
                upper0 = lower0 + self.cropping[0]
                lower1 = (self.input_shape[2]-self.cropping[1])//2
                upper1 = lower1 + self.cropping[1]
            else:
                lower0 = (inputs.shape[1]-self.cropping[0])//2
                upper0 = lower0 + self.cropping[0]
                lower1 = (inputs.shape[2]-self.cropping[1])//2
                upper1 = lower1 + self.cropping[1]

            return inputs[:, lower0:upper0, lower1:upper1, :]

    def get_config(self):
        config = {'cropping': self.cropping, 'data_format': self.data_format}
        base_config = super(CentralCropping2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def from_config(self, config):
        print(config)
        exit()
    # config = {'cropping': self.cropping, 'data_format': self.data_format}
    # base_config = super(CentralCropping2D, self).get_config()
    # return dict(list(base_config.items()) + list(config.items()))
