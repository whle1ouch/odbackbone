from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def res_bottle_neck(inputs, filters, stride=None, name=""):
    """
    resnet v1- residue block from the paper, including 3 convolution respectively '1*1' -> '3*3'->'1*1', and one shortcut between
    input and final convolution output 

    Args:
        input (tf.Tensor): _description_
        filters (int): _description_
        stride (int, tuple(int), None, optional): conv stride.  Defaults to None.
        name (str, optional): _description_. Defaults to "".

    Returns:
        tf.Tensor: output tensor
    """
    channel_in = K.int_shape(inputs)[-1]
    channel_out = filters * 4
    if stride is None:
        stride = 1 if channel_in == channel_out else 2
    h = layers.Conv2D(filters, 1,  strides=stride, padding="same", name=name+"_1_conv")(inputs)
    h = layers.BatchNormalization(epsilon=1.001e-5,  name=name+"_1_bn")(h)
    h = layers.ReLU(name=name+"_1_relu")(h)
    
    h = layers.Conv2D(filters, 3, strides=1, padding="same", name=name+"_2_conv")(h)
    h = layers.BatchNormalization(epsilon=1.001e-5, name=name+"_2_bn")(h)
    h = layers.ReLU(name=name+"_2_relu")(h)
    
    h = layers.Conv2D(channel_out, 1, strides=1, padding="same", name=name+"_3_conv")(h)
    h = layers.BatchNormalization(name=name+"_3_bn")(h)
    
    if channel_in != channel_out:
        shortcut = layers.Conv2D(channel_out, 1, strides=stride, padding="same", name=name+"_0_conv")(inputs)
        shortcut = layers.BatchNormalization(epsilon=1.001e-5, name=name+"_0_bn")(shortcut)
    else:
        shortcut = inputs
        
    h = layers.Add(name=name+"_add")([h, shortcut])
    h = layers.ReLU(name=name+"_out")(h)
    return h
       
def res_bottle_neck_v2(inputs, filters, stride=None, name=""):
    channel_in = K.int_shape(inputs)[-1]
    channel_out = filters * 4
    if stride is None:
        stride = 1
    h = layers.BatchNormalization(epsilon=1.001e-5, name=name + "_preact_bn")(inputs)
    h = layers.ReLU(name=name+"_preact_relu")(h)
    
    if channel_in != channel_out:
        shortcut = layers.Conv2D(channel_out, 1, strides=stride, name=name+"_0_conv")(h)
    else:
        shortcut = layers.MaxPool2D(1, strides=stride)(h) if stride > 1 else inputs
    
    h = layers.Conv2D(filters, 1, strides=1, use_bias=False, name=name+"_1_conv")(h)
    h = layers.BatchNormalization(epsilon=1.001e-5, name=name + "_1_bn")(h)
    h = layers.ReLU(name=name+"_1_relu")(h)
    
    h = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name+"_2_pad")(h)
    h = layers.Conv2D(filters, 3, strides=stride, use_bias=False, name=name+"_2_conv")(h)
    h = layers.BatchNormalization(epsilon=1.001e-5, name=name+"_2_bn")(h)
    h = layers.ReLU(name=name+"_2_relu")(h)
    
    h = layers.Conv2D(channel_out, 1, strides=1, name=name+"_3_conv")(h)
    h = layers.Add(name=name+"_out")([shortcut, h])
    return h
       

def make_res_block(inputs, filters, num_residue, stride=2, name=""):
    output = res_bottle_neck(inputs, filters, stride=stride, name=name+"_block1")
    for i in range(2, num_residue+1):
        output = res_bottle_neck(output, filters, name=name+"_block"+str(i))
    return output

def make_res_block_v2(inputs, filters, num_residue, stride=2, name=""):
    h = res_bottle_neck_v2(inputs, filters, name=name+"_block1")
    for i in range(2, num_residue):
        h = res_bottle_neck_v2(h, filters, name=name+"_block"+str(i))
    h = res_bottle_neck_v2(h, filters, stride=stride, name=name+"_block"+str(num_residue))
    return h

def input_block(inputs, preact=False):
    """
    simple inplement for ResNet input block of reading image data to the network,

    Args:
        inputs (_type_): _description_
        preact (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # resnet_conv1: (3,3) padding -> 7*7*2 conv2d -> bn -> relu -> (1,1) padding -> 3*3*2 avg pool
    output = layers.ZeroPadding2D(3, name="conv1_pad")(inputs)
    output = layers.Conv2D(64, kernel_size=7, strides=2, name="conv1_conv")(output)
    if not preact:
        output = layers.BatchNormalization(epsilon=1.001e-5, name="conv1_bn")(output)
        output = layers.ReLU(name="conv1_relu")(output)
    output = layers.ZeroPadding2D(1, name="pool1_pad")(output)
    output = layers.MaxPool2D(pool_size=3, strides=2, padding="valid", name="pool1_pool")(output)
    return output

def output_block(inputs, output_size, preact, include_top):
    if preact:
        h = layers.BatchNormalization(epsilon=1.001e-5, name="conv1_bn")(inputs)
        h = layers.ReLU(name="conv1_relu")(h)
    else:
        h = inputs
    output = layers.GlobalAveragePooling2D(name="avg_pool")(h)
    logit = layers.Dense(output_size)(output)
    prediction = layers.Softmax()(logit)
    return prediction

def resnet_body(inputs, residue_nums, stride_nums, preact=False):
    assert len(residue_nums) == len(stride_nums), "resudue num is confused"
    n = len(residue_nums)
    init_filters = 64
    h = inputs
    block_func = make_res_block_v2 if preact else make_res_block
    for i in range(n):
        h = block_func(h, init_filters, residue_nums[i], stride_nums[i], name="conv"+str(i+2))
        init_filters *= 2
    return h
    

def Resnet50(output_size=1000, include_top=True, name="resnet50"):
    image_input = layers.Input((224, 224, 3))
    output = input_block(image_input)
    output = resnet_body(output, [3, 4, 6, 3], [1, 2, 2, 2], False)
    prediction = output_block(output, output_size, False, include_top)
    model = Model(image_input, prediction, name=name)
    return model

def Resnet50V2(output_size=1000, include_top=True, name="resnet50v2"):
    image_input = layers.Input((224, 224, 3))
    output = input_block(image_input, True)
    output = resnet_body(output, [3, 4, 6, 3], [2, 2, 2, 1], True)
    prediction = output_block(output, output_size, True, include_top)
    model = Model(image_input, prediction, name=name)
    return model

def Resnet101(output_size=1000, include_top=True, name="resnet101"):
    image_input = layers.Input((224, 224, 3))
    output = input_block(image_input)
    output = resnet_body(output, [3, 4, 23, 3], [1, 2, 2, 2], False)
    prediction = output_block(output, output_size, False, include_top)
    model = Model(image_input, prediction, name=name)
    return model

def Resnet101V2(output_size=1000, include_top=True, name="resnet101v2"):
    image_input = layers.Input((224, 224, 3))
    output = input_block(image_input, True)
    output = resnet_body(output, [3, 4, 23, 3], [2, 2, 2, 1], True)
    prediction = output_block(output, output_size, True, include_top)
    model = Model(image_input, prediction, name=name)
    return model

def Resnet152(output_size=1000, include_top=True, name="resnet152"):
    image_input = layers.Input((224, 224, 3))
    output = input_block(image_input)
    output = resnet_body(output, [3, 8, 36, 3], [1, 2, 2, 2], False)
    prediction = output_block(output, output_size, False, include_top)
    model = Model(image_input, prediction, name=name)
    return model

def Resnet152V2(output_size=1000, include_top=True, name="resnet152v2"):
    image_input = layers.Input((224, 224, 3))
    output = input_block(image_input, True)
    output = resnet_body(output, [3, 8, 36, 3], [2, 2, 2, 1], True)
    prediction = output_block(output, output_size, True, include_top)
    model = Model(image_input, prediction, name=name)
    return model
    
    


if __name__ == "__main__":
    model = Resnet152V2()
    model.summary()
    print("------------------build succuss -----------------------")