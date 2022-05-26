from tensorflow.keras import layers
from tensorflow.keras.models import Model


def conv2d_bn(inputs, filters, kernel, strides=(1, 1), padding="same", name="convbn"):
    h = layers.Conv2D(filters, kernel, strides, padding=padding, use_bias=False, name=name+"_conv")(inputs)
    h = layers.BatchNormalization(epsilon=1.001e-5, scale=False, name=name+"_bn")(h)
    h = layers.ReLU(name=name+"_relu")(h)
    return h

def input_block(inputs):
    h = conv2d_bn(inputs, 32, 3, strides=(2, 2), padding='valid', name="Conv1")
    h = conv2d_bn(h, 32, 3, padding='valid', name="Conv2")
    h = conv2d_bn(h, 64, 3, name="Conv3")
    h = layers.MaxPooling2D((3, 3), strides=(2, 2))(h)

    h = conv2d_bn(h, 80, 1, padding='valid', name="Conv4")
    h = conv2d_bn(h, 192, 3, padding='valid', name="Conv5")
    h = layers.MaxPooling2D((3, 3), strides=(2, 2))(h)
    return h 


def inception_block_v3_1(inputs, filters_1, filters_2_reduce,  filters_2, filters_3_reduce, filters_3, filters_4, stride, name):
    """
    the first block of inception v3, containing 4 pathes blow:
    1. one 1x1 conv, 
    2. one 1x1 conv and one 5x5 conv, or one 3x3 conv
    3. 2 times 3x3 conv,
    4. avg pool and a 1x1 conv.

    Args:
        inputs (tf.Tensor): input tensor
        filters_1 (int): filters of 1x1 conv in path 1 
        filters_2_reduce (int): filters of 1x1 conv in path 2
        filters_2 (int): filters of 5x5 or 3x3 conv in path 2
        filters_3_reduce (int): filters of first 3x3 conv in path 3
        filters_3 (int): filters of second 3x3 conv in path 3
        filters_4 (int): filters of 1x1 conv in path 4
        stride (int): stride of the final convolution(pool) in each pathes
        name (str): prefix of block name

    Returns:
        tf.Tensor: concat output of each pathes 
    """
    hs = []
    padding = "valid" if stride == 2 else "same"
    if filters_1 > 0:
        h1 = conv2d_bn(inputs, filters_1, 1, stride, padding=padding, name=name+"_p1_conv1")
        hs.append(h1)
    
    if filters_2_reduce > 0:
        h2 = conv2d_bn(inputs, filters_2_reduce, 1, 1, name=name+"_p2_conv1")
        h2 = conv2d_bn(h2, filters_2, 5, stride, name=name+"_p2_conv2")
    else:
        h2 = conv2d_bn(inputs, filters_2, 3, stride, padding=padding, name=name+"_p2_conv1")
    hs.append(h2)
    
    h3 = conv2d_bn(inputs, filters_3_reduce, 1, 1, name=name+"_p3_conv1")
    h3 = conv2d_bn(h3, filters_3, 3, 1, name=name+"_p3_conv2")
    h3 = conv2d_bn(h3, filters_3, 3, stride, padding=padding, name=name+"_p3_conv3")
    hs.append(h3)
    
    h4 = layers.MaxPooling2D((3, 3), stride, padding=padding, name=name+"_p4_pool")(inputs)
    if filters_4 > 0:
        h4 = conv2d_bn(h4, filters_4, 1, 1, name=name+"_p4_conv1")
    hs.append(h4)
    
    h = layers.Concatenate(name=name+"_concat")(hs)
    return h


def inception_block_v3_2(inputs, filters_1, filters_2_reduce,  filters_2, filters_3_reduce, filters_3, filters_4, stride, n, name):
    """
    the second block of inception v3, also containing 4 pathes little different from block 1

    Args:
        inputs (tf.Tensor): input tensor
        filters_1 (int): filters of 1x1 conv in path 1 
        filters_2_reduce (int): filters of 1x1 conv in path 2
        filters_2 (int): filters of last conv in path 2
        filters_3_reduce (int): filters of first 3x3 conv in path 3
        filters_3 (int): filters of last conv in path 3
        filters_4 (int): filters of 1x1 conv in path 4
        stride (int): stride of the final convolution(pool) in each pathes
        n (int): convolution kernel size in path 2 and 3
        name (str): prefix of block name

    Returns:
        tf.Tensor: concat output of each pathes 
    """
    hs = []
    if stride == 1:
        h1 = conv2d_bn(inputs, filters_1, 1, stride, name=name+"_p1_conv1")
        hs.append(h1)
        
        h2 = conv2d_bn(inputs, filters_2_reduce, 1, name=name+"_p2_conv1")
        h2 = conv2d_bn(h2, filters_2_reduce, (1, n), name=name+"_p2_conv2")
        h2 = conv2d_bn(h2, filters_2, (n, 1), name=name+"_p2_conv3")
        hs.append(h2)
        
        h3 = conv2d_bn(inputs, filters_3_reduce, 1, name=name+"_p3_conv1")
        h3 = conv2d_bn(h3, filters_3_reduce, (n, 1), name=name+"_p3_conv2")
        h3 = conv2d_bn(h3, filters_3_reduce, (1, n), name=name+"_p3_conv3")
        h3 = conv2d_bn(h3, filters_3_reduce, (n, 1), name=name+"_p3_conv4")
        h3 = conv2d_bn(h3, filters_3, (1, n), name=name+"_p3_conv5")
        hs.append(h3)
        
        h4 = layers.MaxPooling2D((3, 3), 1, padding="same", name=name+"_p4_pool")(inputs)
        h4 = conv2d_bn(h4, filters_4, 1, name=name+"_p4_conv1")
        hs.append(h4)
    else:
        h2 = conv2d_bn(inputs, filters_2_reduce, 1, name=name+"_p1_conv1")
        h2 = conv2d_bn(h2, filters_2, 3, 2, padding="valid", name=name+"_p1_conv2")
        hs.append(h2)
        
        h3 = conv2d_bn(inputs, filters_3_reduce, 1, name=name+"_p2_conv1")
        h3 = conv2d_bn(h3, filters_3_reduce, (1, n), name=name+"_p2_conv2")
        h3 = conv2d_bn(h3, filters_3_reduce, (n, 1), name=name+"_p2_conv3")
        h3 = conv2d_bn(h3, filters_3, 3, 2, padding="valid", name=name+"_p2_conv4")
        hs.append(h3)
        
        h4 = layers.MaxPooling2D((3, 3), (2, 2), name=name+"_p3_pool")(inputs)
        hs.append(h4)
    h = layers.Concatenate(name=name+"_concat")(hs)
    return h

def inception_block_v3_3(inputs, filters_1, filters_2_reduce,  filters_2, filters_3_reduce, filters_3, filters_4, stride, name):
    """
    the third block of inception v3, used in final stage the maximum feature extraction.
    Args:
        inputs (tf.Tensor): input tensor
        filters_1 (int): filters of 1x1 conv in path 1 
        filters_2_reduce (int): filters of 1x1 conv in path 2
        filters_2 (int): filters of 5x5 or 3x3 conv in path 2
        filters_3_reduce (int): filters of first 3x3 conv in path 3
        filters_3 (int): filters of second 3x3 conv in path 3
        filters_4 (int): filters of 1x1 conv in path 4
        stride (int): stride of the final convolution(pool) in each pathes
        name (str): prefix of block name

    Returns:
        tf.Tensor: concat output of each pathes 
    """
    hs = []
    h1 = conv2d_bn(inputs, filters_1, 1, stride, name=name+"_p1_conv1")
    hs.append(h1)
    
    h2 = conv2d_bn(inputs, filters_2_reduce, 1, name=name+"_p2_conv1")
    h21 = conv2d_bn(h2, filters_2, (1, 3), name=name+"_p2_conv2")
    h22 = conv2d_bn(h2, filters_2, (3, 1), stride, name=name+"_p2_conv3")
    h2 = layers.Concatenate(name=name+"_conv2_concat")([h21, h22])
    hs.append(h2)
    
    
    h3 = conv2d_bn(inputs, filters_3_reduce, 1, name=name+"_p3_conv1")
    h3 = conv2d_bn(h3, filters_3, 3, name=name+"_p3_conv2")
    h31 = conv2d_bn(h3, filters_3, (1, 3), stride, name=name+"_p3_conv3")
    h32 = conv2d_bn(h3, filters_3, (3, 1), stride, name=name+"_p3_conv4")
    h3 = layers.Concatenate(name=name+"_conv3_concat")([h31, h32])
    hs.append(h3)
    
    h4 = layers.MaxPooling2D((3, 3), stride, padding="same", name=name+"_p4_pool")(inputs)
    h4 = conv2d_bn(h4, filters_4, 1, 1, name=name+"_p4_conv1")
    hs.append(h4)
    h = layers.Concatenate(name=name+"_concat")(hs)
    return h

def output_block(inputs, output_size, include_top):
    h = layers.GlobalAveragePooling2D()(inputs)
    if include_top:
        h = layers.Dense(output_size, name="logit")(h)
        h = layers.Softmax(name="prediction")(h)
    return h
        
def InceptionV3(output_size=1000, include_top=True, name="InceptionV3"):
    image_inputs = layers.Input((299, 299, 3))
    outputs = input_block(image_inputs)
    
    # 4 times block 1
    outputs = inception_block_v3_1(outputs, 64, 48, 64, 64, 96, 32, 1, name="mix0")
    outputs = inception_block_v3_1(outputs, 64, 48, 64, 64, 96, 64, 1, name="mix1")
    outputs = inception_block_v3_1(outputs, 64, 48, 64, 64, 96, 64, 1, name="mix2")
    outputs = inception_block_v3_1(outputs, 0, 0, 384, 64, 96, 0, 2, name="mix3")
    # 5 times block 2
    outputs = inception_block_v3_2(outputs, 192, 128, 192, 128, 192, 192, 1, 7, name="mix4")
    outputs = inception_block_v3_2(outputs, 192, 160, 192, 160, 192, 192, 1, 7, name="mix5")
    outputs = inception_block_v3_2(outputs, 192, 160, 192, 160, 192, 192, 1, 7, name="mix6")
    outputs = inception_block_v3_2(outputs, 192, 192, 192, 192, 192, 192, 1, 7, name="mix7")
    outputs = inception_block_v3_2(outputs, 0, 192, 320, 192, 192, 0, 2, 7, name="mix8")
    # 2 times block 3
    outputs = inception_block_v3_3(outputs, 320, 384, 384, 448, 384, 192, 1, name="mix9")
    outputs = inception_block_v3_3(outputs, 320, 384, 384, 448, 384, 192, 1, name="mix10")
    
    pred = output_block(outputs, output_size, include_top)
    model = Model(image_inputs, pred, name=name)
    return model

if __name__ == "__main__":
    model = InceptionV3()
    model.summary()