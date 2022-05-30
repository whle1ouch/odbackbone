from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from functools import wraps, reduce

def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), funcs)
    else:
        raise ValueError("Composition of empty sequence not supported.")

def mish(inputs):
    return inputs * K.tanh(K.softplus(inputs))

class Mish(layers.Layer):
    
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.supports_masking = True
    
    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))
    
    def get_config(self):
        return super().get_config()
    
    def compute_output_shape(self, input_shape):
        return input_shape

@wraps(layers.Conv2D)
def DarkNetConv2D(filters, kernel, strides, use_bias=True):
    if strides == (2, 2) or strides == 2:
        padding = "valid"
    else:
        padding = "same"
    return layers.Conv2D(filters, kernel, strides, padding=padding, use_bias=use_bias, kernel_regularizer=l2(5e-4))

def DarknetConv2D_BN_Mish(filters, kernel, strides=(1, 1)):
    return compose(
        DarkNetConv2D(filters, kernel, strides, use_bias=False),
        layers.BatchNormalization(),
        Mish()
    )
    
def csp_res_block(inputs, filters, num_blocks, all_narrow=True):
    """
    the main block of cspdarknet, including a down-sample convolution , `num_blocks` times residue connection block
    and a long shortcut residue connect

    Args:
        inputs (tf.Tensor): input tensor
        filters (int): output filter nums
        num_blocks (int): resnet-like block's repeat time
        all_narrow (bool, optional): weather to reduces the width of blocks. Defaults to True.

    Returns:
        tf.Tensor: output tensor
    """
    pre = layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
    pre = DarknetConv2D_BN_Mish(filters, 3, 2)(pre)
    
    shortcut = DarknetConv2D_BN_Mish(filters // 2 if all_narrow else filters, 1)(pre)
    
    h = DarknetConv2D_BN_Mish(filters // 2 if all_narrow else filters, 1)(pre)
    
    for _ in range(num_blocks):
        h1 = DarknetConv2D_BN_Mish(filters // 2, 1)(h)
        h1 = DarknetConv2D_BN_Mish(filters // 2 if all_narrow else filters, 3)(h1)
        h = layers.Add()([h, h1])
    
    h = DarknetConv2D_BN_Mish(filters // 2 if all_narrow else filters, 1)(h)
    h = layers.Concatenate()([h, shortcut])
    h = DarknetConv2D_BN_Mish(filters, 1)(h)
    return h

def cspdarknet53(inputs):
    """
    the cspdarknet53 body, backbone of YOLO V4, including 5 times down-sampling to increase the receptive field 
    of each pixels in feature map of outputm, 

    Args:
        inputs (tf.Tensor): input tensor

    Returns:
        tf.Tensor, tf.Tensor, tf.Tensor: 3 output of the feature pyramids
    """
    h = DarknetConv2D_BN_Mish(32, 3)(inputs)
    h = csp_res_block(h, 64, 1, False)
    h = csp_res_block(h, 128, 2)
    h = csp_res_block(h, 256, 8)
    feat1 = h
    h = csp_res_block(h, 512, 8)
    feat2 = h
    h = csp_res_block(h, 1024, 4)
    feat3 = h
    return feat1, feat2, feat3


def test_cspdarknet53():
    inputs = layers.Input((416, 416, 3))
    h1, h2, h3 = cspdarknet53(inputs)
    model = Model(inputs, [h1, h2, h3], name="darknet53")
    model.summary()
    
if __name__ == "__main__":
    test_cspdarknet53()