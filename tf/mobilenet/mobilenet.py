from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

    
def input_block(inputs, filters, stride):
    h = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, name="conv1")(inputs)
    h = layers.BatchNormalization(epsilon=1.001e-5, name="conv1_bn")(h)
    h = layers.ReLU(max_value=6., name="conv1_relu6")(h)
    return h
    
def output_block(inputs, output_size, include_top):
    if include_top:
        h = layers.GlobalAveragePooling2D(name="global_avg_pool")(inputs)
        filters = h.shape[-1]
        h = layers.Reshape((1, 1, filters), name="reshape_1")(h)
        h = layers.Dropout(rate=0.2, name="dropout")(h)
        h = layers.Conv2D(output_size, 1, name="conv_preds")(h)
        h = layers.Reshape((output_size,), name="reshape_2")(h)
        h = layers.Softmax(name="predictions")(h)
    else:
        h = layers.GlobalAveragePooling2D()(inputs)
    return h

def output_block_v2(inputs, output_size,  last_block_filters, include_top):
    if last_block_filters is None:
        last_block_filters = 1280
    h = layers.Conv2D(1280, 1, 1, use_bias=False, name="conv_1")(inputs)
    h = layers.BatchNormalization(epsilon=1.001e-5, name="conv_1_bn")(h)
    h = layers.ReLU(max_value=6., name="out_relu")(h)
    if include_top:
        h = layers.GlobalAveragePooling2D(name="global_avg_pool")(h)
        h = layers.Dense(output_size, name="predictions")(h)
    else:
        h = layers.GlobalAveragePooling2D()(h)
    return h
    
def mobilenet_block(inputs, filters, stride, block_id=None):
    """
    bottleneck block of mobile net, including a depthwise convolution and a custom 1x1 convolution, 
    that could reduce the parameters.

    Args:
        inputs (tf.Tensor): input tensor
        filters (int): output filters
        stride (int or tuple(int, int)): depthwise convolution's stride
        block_id (int or None): block id used in block name's prefix

    Returns:
        tf.Tensor: output
    """
    if block_id:
        block_title = str(block_id)
    else:
        block_title = "0"
    h = layers.DepthwiseConv2D(3, stride, padding="SAME", use_bias=False, name="conv_dw_"+block_title)(inputs)
    h = layers.BatchNormalization(epsilon=1.001e-5, name="conv_dw_"+block_title+"_bn")(h)
    h = layers.ReLU(max_value=6., name="conv_dw_"+block_title+"_relu6")(h)
    h = layers.Conv2D(filters, 1, use_bias=False, name="conv_pw_"+block_title)(h)
    h = layers.BatchNormalization(epsilon=1.001e-5, name="conv_pw_"+block_title+"_bn")(h)
    h = layers.ReLU(max_value=6., name="conv_pw_"+block_title+"_relu6")(h)
    return h

def mobilenet_block_v2(inputs, filters, stride, block_id=0):
    """
    bottleneck block of mobile net v2, the main difference from v1 version is introducing the residue connection.

    Args:
        inputs (tf.Tensor): input tensor
        filters (int): output filters
        stride (int or tuple(int, int)): depthwise convolution's stride
        block_id (int or None): block id used in block name's prefix

    Returns:
        tf.Tensor: output
    """
    channel_in = K.int_shape(inputs)[-1]
    if block_id: 
        prefix = f"block_{block_id}"
        h = layers.Conv2D(6 * channel_in, 1, use_bias=False, name=prefix+"_expand")(inputs)
        h = layers.BatchNormalization(epsilon=1.001e-5, name=prefix+"_expand_bn")(h)
        h = layers.ReLU(max_value=6, name=prefix+"_expand_relu6")(h)
    else:
        prefix = "expand_conv"
        h = inputs
    
    h = layers.ZeroPadding2D(padding=(1, 1), name=prefix+"_pad")(h)
    h = layers.DepthwiseConv2D(3, strides=stride, use_bias=False, name=prefix+"_dw")(h)
    h = layers.BatchNormalization(epsilon=1.001e-5, name=prefix+"_dw_bn")(h)
    h = layers.ReLU(max_value=6, name=prefix+"_dw_relu6")(h)
    
    h = layers.Conv2D(filters, 1, use_bias=False, padding="same", name=prefix+"_project")(h)
    h = layers.BatchNormalization(epsilon=1.001e-5, name=prefix+"_project_bn")(h)
    
    if stride == 1 and channel_in == filters:
        h = layers.Add(name=prefix+"_add")([h, inputs])
    return h


def Mobilenet(output_size=1000, include_top=True, name="MobileNet"):
    """
    a simple inplement of mobilenet, the filters and strides are all from the paper

    Args:
        output_size (int): output size for image classifation. Defaults to 1000.
        name (str): netword name. Defaults to "MobileNet".

    Returns:
        _type_: _description_
    """
    filters = [32, 64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2
    strides = [2, 1, 2, 1, 2, 1, 2] + [1] * 5 + [2, 1]
    n = len(filters)
    inputs = layers.Input((224, 224, 3), name="input_1")
    output = input_block(inputs, filters[0], strides[0])
    for i in range(1, n):
        output = mobilenet_block(output, filters[i], strides[i], block_id=i)
    pred = output_block(output, output_size, include_top)
    model = Model(inputs, pred, name=name)
    return model
        
    
def MobilenetV2(output_size=1000, include_top=True, name="MobileNetV2"):
    """
    a simple inplement of mobilenet v2, the filters and strides are all from the paper

    Args:
        output_size (int, optional): _description_. Defaults to 1000.
        name (str, optional): _description_. Defaults to "MobileNetV2".
    """
    filters = [32, 16] + [24]*2 + [32] * 3  + [64] * 4  + [96] * 3 + [160] * 3 + [320]    
    strides = [2, 1]  + [2, 1] + [2, 1, 1] + [2,1,1,1] + [1] * 3  + [2, 1, 1] + [1]
    n = len(filters)
    inputs = layers.Input((224, 224, 3), name="input_1")
    output = input_block(inputs, filters[0], strides[0])
    for i in range(1, n):
        output = mobilenet_block_v2(output, filters[i], strides[i], block_id=i-1)
    pred = output_block_v2(output, output_size, 1280, include_top)
    model = Model(inputs, pred, name=name)
    return model
        

if __name__  == "__main__":
    model = MobilenetV2()
    model.summary()