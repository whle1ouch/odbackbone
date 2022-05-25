from tensorflow.keras import layers
from tensorflow.keras.models import Model


def conv2d_bn(inputs, filters, kernel, stride=1, padding="valid", name=""):
    h = layers.Conv2D(filters, kernel, stride, padding=padding, name=name+"_conv")(inputs)
    h = layers.BatchNormalization(epsilon=1.001e-5, name=name+"_bn")(h)
    h = layers.ReLU(name=name+"_relu")(h)
    return h
    
def input_block(inputs):
    h = conv2d_bn(inputs, 64, 7, 2, padding="same", name="Conv1")
    h = layers.MaxPooling2D((3, 3), 2, padding="same", name="Pool_1")(h)
    h = conv2d_bn(h, 64, 1, 1, name="Conv2")
    h = conv2d_bn(h, 192, 3, 1, padding="same", name="Conv3")
    h = layers.MaxPooling2D((3, 3), 2, padding="same", name="Pool_2")(h)
    return h


def inception_block_v2(inputs, filters_1, filters_3_reduce,  filters_3, filters_3x2_reduce, filters_3x2, filters_pool, stride, name):
    """
    a inception block improved from inception v1, the main trick is using two 3x3 convolution to replace the 5x5 conv for reducing parameters,
    and also introducing batch normalization to regularize model

    Args:
        inputs (tf.Tensor): input tensor
        filters_1 (int): filter nums in 1x1 conv path
        filters_3_reduce (int): filter nums of 1x1 conv in 3x3 conv path, which can squeeze the channels
        filters_3 (int): filter nums of 3x3 conv in 3x3 conv path
        filters_3x2_reduce (int): filter nums of 1x1 conv in 5x5 conv path, which can squeeze the channels
        filters_3x2 (int): filter nums of 5x5 conv in 5x5 conv path
        filters_pool (int): filter nums of 1x1 conv in max pool
        stride (int): the stride of finnal stride in each path, which is used to control map size.
        name (str): prefix of block name

    Returns:
        tf.Tensor: concat output of each path by channel axis
    """
    hs = []
    if filters_1 > 0:
        h1 = conv2d_bn(inputs, filters_1, 1, stride, padding="same", name=name+"_1x1")
        hs.append(h1)
    
    h2 = conv2d_bn(inputs, filters_3_reduce, 1, 1, padding="same", name=name+"_3x3_1")
    h2 = conv2d_bn(h2, filters_3, 3, stride, padding="same", name=name+"_3x3_2")
    hs.append(h2)
    
    h3 = conv2d_bn(inputs, filters_3x2_reduce, 1, 1, padding="same", name=name+"_5x5_1")
    h3 = conv2d_bn(h3, filters_3x2, 3, 1, padding="same", name=name+"_5x5_2")
    h3 = conv2d_bn(h3, filters_3x2, 3, stride, padding="same", name=name+"_5x5_3")
    hs.append(h3)
    
    h4 = layers.MaxPooling2D((3, 3), stride, padding="same", name=name+"_pool")(inputs)
    if filters_pool > 0: 
        h4 = conv2d_bn(h4, filters_pool, 1, 1, padding="same", name=name+"_pool_conv")
    hs.append(h4)
    
    h = layers.Concatenate(name=name+"_concat")(hs)
    return h


def output_block(inputs, output_size, include_top):
    h = layers.GlobalAveragePooling2D()(inputs)
    if include_top:
        h = layers.Dense(output_size, name="logit")(h)
        h = layers.Softmax(name="prediction")(h)
    return h

def InceptionV2(output_size=1000, include_top=True, name="InceptionV2"):
    """
    this is a simple implement of inception v2, not from paper however, but from the source code of tensorflow v1.0,
    see this in https://github.com/leimao/DeepLab-V3/blob/master/nets/inception_v2.py

    Args:
        output_size (int, optional): _description_. Defaults to 1000.
        include_top (bool, optional): _description_. Defaults to True.
        name (str, optional): _description_. Defaults to "InceptionV2".

    Returns:
        _type_: _description_
    """
    inputs = layers.Input((224, 224, 3))
    h = input_block(inputs)
    
    h = inception_block_v2(h, 64, 64, 64, 64, 96, 32, 1, "3a")
    h = inception_block_v2(h, 64, 64, 64, 64, 96, 32, 1, "3b")
    h = inception_block_v2(h, 0, 128, 160, 64, 96, 0, 2, "3c")
    
    h = inception_block_v2(h, 224, 64, 96, 96, 128, 128, 1, "4a")
    h = inception_block_v2(h, 192, 96, 128, 96, 128, 128, 1, "4b")
    h = inception_block_v2(h, 160, 128, 160, 128, 160, 96, 1, "4c")
    h = inception_block_v2(h, 96, 128, 192, 160, 192, 96, 1, "4d")
    h = inception_block_v2(h, 0, 128, 192, 192, 256, 0, 2, "4e")

    h = inception_block_v2(h, 352, 192, 320, 160, 224, 128, 1, "5a")
    h = inception_block_v2(h, 352, 192, 320, 192, 224, 128, 1, "5b")
    
    output = output_block(h, output_size, include_top)
    model = Model(inputs, output, name=name)
    return model



if __name__ == "__main__":
    model = InceptionV2()
    model.summary()   