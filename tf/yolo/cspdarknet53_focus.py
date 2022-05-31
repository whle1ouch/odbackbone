from xml.sax.xmlreader import InputSource
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from functools import reduce
import math

def compose_func(f1, f2):
    def wrapper(*args, **kwargs):
        return f2(f1(*args, **kwargs))
    return wrapper

def compose(*funcs):
    return reduce(compose_func, funcs)

class SiLU(layers.Layer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_masking = True
    
    def call(self, inputs):
        return inputs * K.sigmoid(inputs)
    
    def get_config(self):
        return super().get_config()

    def compute_output_shape(self, input_shape):
        return input_shape

class Focus(layers.Layer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def call(self, inputs):
        h, w = K.int_shape(inputs)[1:3]
        h_pad = (0, 0) if h % 2 == 0 else (1, 0)
        w_pad = (0, 0) if w % 2 == 0 else (1, 0)
        if h_pad == (0, 0) and w_pad == (0, 0):
            x = inputs
        else:
            x = K.spatial_2d_padding(inputs, padding=(h_pad, w_pad))
        x = K.concatenate([x[..., ::2, ::2, :], x[..., 1::2, 1::2, :], x[..., ::2, 1::2, :], x[..., 1::2, ::2, :]], axis=-1)
        return h
        
    def get_config(self):
        return super().get_config()
    
    def compute_output_shape(self, input_shape):
        h, w = input_shape[1: 3]
        hp = h // 2 if h % 2 == 0 else h // 2 + 1
        wp = w // 2 if w % 2 == 0 else w // 2 + 1
        return (input_shape[0], hp, wp, input_shape[-1] * 4)

def DarknetConv2D(filters, kernel, strides=(1, 1), padding=None, use_bias=True):
    if padding is None:
        padding = "same"
    return compose(layers.Conv2D(filters, kernel, strides, padding=padding, use_bias=use_bias, kernel_regularizer=l2(5e-4)),
                   layers.BatchNormalization(),
                   SiLU())


def focus(inputs):
    h, w = K.int_shape(inputs)[1:3]
    h_pad = (0, 0) if h % 2 == 0 else (1, 0)
    w_pad = (0, 0) if w % 2 == 0 else (1, 0)
    if h_pad == (0, 0) and w_pad == (0, 0):
        x = inputs
    else:
        x = layers.ZeroPadding2D((h_pad, w_pad))(inputs)
    return layers.Concatenate(axis=-1)([x[..., ::2, ::2, :], x[..., 1::2, 1::2, :], x[..., ::2, 1::2, :], x[..., 1::2, ::2, :]])


def csp_bottleneck(inputs, filters, num_blocks, shortcut=True, expand=0.5):
    filters_ = int(filters * expand)
    s = DarknetConv2D(filters_, 1)(inputs)
    h = DarknetConv2D(filters_, 1)(inputs)
    for _ in range(num_blocks):
        h1 = DarknetConv2D(filters_, 1)(h)
        h1 = DarknetConv2D(filters_, 3)(h1)
        if shortcut:
            h = layers.Add()([h, h1])
        else:
            h = h1
    h = layers.Concatenate()([h, s])
    h = DarknetConv2D(filters, 1)(h)
    return h 

def sppf(inputs, filters, pool_size=5):
    filters_ = K.int_shape(inputs)[-1] // 2
    h1 = DarknetConv2D(filters_, 1)(inputs)
    
    h2 = layers.ZeroPadding2D(pool_size // 2)(h1)
    h2 = layers.MaxPooling2D(pool_size, strides=1)(h2)
    
    h3 = layers.ZeroPadding2D(pool_size // 2)(h2)
    h3 = layers.MaxPooling2D(pool_size, strides=1)(h3)
    
    h4 = layers.ZeroPadding2D(pool_size // 2)(h3)
    h4 = layers.MaxPooling2D(pool_size, strides=1)(h4)
    h = layers.Concatenate()([h1, h2, h3, h4])
    h = DarknetConv2D(filters, 1)(h)
    return h
    
def depth_gain(n, depth_multiple):
    return max(round(n * depth_multiple), 1) if n > 1 else n

def width_gain(n, divisor=8):
    return int(math.ceil(n / divisor) * divisor)

def _cspdarknet(inputs, depth_multiple, width_multiple):
    h = layers.ZeroPadding2D(2)(inputs)
    h = DarknetConv2D(width_gain(64 * width_multiple), 6, 2, "valid")(h)
    h = DarknetConv2D(width_gain(128 * width_multiple), 3, 2)(h)
    h = csp_bottleneck(h, width_gain(128 * width_multiple), depth_gain(3, depth_multiple))
    h = DarknetConv2D(width_gain(256 * width_multiple), 3, 2)(h)
    h = csp_bottleneck(h, width_gain(256 * width_multiple), depth_gain(6, depth_multiple))
    feat1 = h
    h = DarknetConv2D(width_gain(512 * width_multiple), 3, 2)(h)
    h = csp_bottleneck(h, width_gain(512 * width_multiple), depth_gain(9, depth_multiple))
    feat2 = h
    h = DarknetConv2D(width_gain(1024 * width_multiple), 3, 2)(h)
    h = csp_bottleneck(h, width_gain(1024 * width_multiple), depth_gain(3, depth_multiple))
    h = sppf(h, width_gain(1024 * width_multiple), 5)
    feat3 = h
    return feat1, feat2, feat3


def CspDarkNetSmall(inputs):
    return _cspdarknet(inputs, 0.33, 0.50)

def CspDarkNetXtreme(inputs):
    return _cspdarknet(inputs, 1.33, 1.25)

def CspDarkNetLarge(inputs):
    return _cspdarknet(inputs, 1.0, 1.0)

def CspDarkNetMedium(inputs):
    return _cspdarknet(inputs, 0.67, 0.75)

def CspDarkNetNano(inputs):
    return _cspdarknet(inputs, 0.33, 0.25)




if __name__ == "__main__":
    inputs = layers.Input((608, 608, 3))
    h1, h2, h3 = CspDarkNetLarge(inputs)
    model = Model(inputs, [h1, h2, h3], name="cspdarknet")
    model.summary()