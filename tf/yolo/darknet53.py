from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from functools import wraps, reduce


def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), funcs)
    else:
        raise ValueError("Composition of empty sequence not supported.")


@wraps(layers.Conv2D)
def DarkNetConv2D(filters, kernel, strides, use_bias=True):
    if strides == (2, 2) or strides == 2:
        padding = "valid"
    else:
        padding = "same"
    return layers.Conv2D(filters, kernel, strides, padding=padding, use_bias=use_bias, kernel_regularizer=l2(5e-4))

def DarknetConv2D_BN_Leaky(filters, kernel, strides=(1, 1)):
    return compose(
        DarkNetConv2D(filters, kernel, strides, use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1)
    )

def dark_res_block(inputs, filters, num_blocks):
    """
    the main block of darknet, including a down-sample convolution and resnet-like struct of `num_blocks`

    Args:
        inputs (tf.Tensor): input tensor
        filters (int): output filter nums
        num_blocks (int): resnet-like block's repeat time

    Returns:
        tf.Tensor: output tensor
    """
    h = layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
    h = DarknetConv2D_BN_Leaky(filters, 3, 2)(h)
    for _ in range(num_blocks):
        h1 = DarknetConv2D_BN_Leaky(filters // 2, 1)(h)
        h1 = DarknetConv2D_BN_Leaky(filters, 3)(h1)
        h = layers.Add()([h, h1])
    return h

def darnet53(inputs):
    """
    the darknet53 body, backbone of YOLO V3, including 5 times down-sampling to increase the receptive field of each pixels in feature map of output
    Args:
        inputs (tf.Tensor): input tensor

    Returns:
        tf.Tensor, tf.Tensor, tf.Tensor: 3 output of the feature pyramids
    """
    h = DarknetConv2D_BN_Leaky(32, 3)(inputs)
    h = dark_res_block(h, 64, 1)
    h = dark_res_block(h, 128, 2)
    h = dark_res_block(h, 256, 8)
    feat1 = h
    h = dark_res_block(h, 512, 8)
    feat2 = h
    h = dark_res_block(h, 1024, 4)
    feat3 = h
    return feat1, feat2, feat3


def test_darknet():
    inputs = layers.Input((416, 416, 3))
    h1, h2, h3 = darnet53(inputs)
    model = Model(inputs, [h1, h2, h3], name="darknet53")
    model.summary()


if __name__ == "__main__":
    test_darknet()    