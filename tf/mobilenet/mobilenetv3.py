from tkinter import _Padding
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import applications
from torch import dropout

def hard_swish(x):
  return layers.Multiply()([K.hard_sigmoid(x), x])
    
def get_depth(v, divisor=8, min_value=None):
    """
    get the depth(filter nums) of depth-wise convolution layers or expand layers, this function will get the divisor times
    which is around v (a little larger than v).
     

    Args:
        v (_type_): _description_
        divisor (int, optional): _description_. Defaults to 8.
        min_value (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
    
def se_block(inputs, filters, se_ratio, prefix):
    h = layers.GlobalAveragePooling2D(name=prefix+"squeeze_exite/AvgPool")(inputs)
    h = layers.Reshape((1, 1, filters))(h)
    h = layers.Conv2D(get_depth(filters*se_ratio), 1, 
                      strides=1, 
                      use_bias=False, 
                      padding="same",
                      name=prefix+"squeeze_excite/Conv")(h)
    h = layers.ReLU(name=prefix+"squeeze_excite/Relu")(h)
    h = layers.Conv2D(filters, 1, padding="same", name=prefix+"squeeze_excite/Conv_1")(h)
    h = K.hard_sigmoid(h)
    h = layers.Multiply(name=prefix+"squeeze_excite/Mul")([inputs, h])
    return h

def input_block(inputs, filters, stride, activation):
    h = layers.Conv2D(
        filters, 3,
        strides=stride,
        padding="same",
        use_bias=False, 
        name="Conv"
    )(inputs)
    h = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name="Conv/BatchNorm")(h)
    h = activation(h)
    return h   
    
def mobilenet_block_v3(inputs, filters, kernel_size, stride, se_ratio, activation, block_id):
    channel_in = K.int_shape(inputs)[-1]
    shortcut = inputs
    prefix = 'expanded_conv/'
    if block_id:
        prefix = 'expanded_conv_{}/'.format(block_id)
        h = layers.Conv2D(get_depth(channel_in * 6), 
                          1, 
                          padding="same", 
                          use_bias=False, 
                          name=prefix+"expand")(h)
        h = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix+"expand/BatchNorm")(h)
        h = activation(h)
    else:
        h = inputs
    
    if stride == 2:
        input_size = K.int_shape(h)[-3: -1]
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
        correct = (kernel_size[0]//2, kernel_size[1]//2)
        pad = ((correct[0]- adjust[0], correct[0]), (correct[1]-adjust[1], correct[1]))
        h =  layers.ZeroPadding2D(padding=pad, name=prefix+"depthwise/pad")(h)
    h = layers.DepthwiseConv2D(kernel_size, 
                               strides=stride, 
                               padding="same" if stride == 1 else "valid", 
                               use_bias=False, 
                               name=prefix+"depthwise")(h)
    h = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix+"depthwise/BatchNorm")(h)
    h = activation(h)
    
    if se_ratio:
        h = se_block(h, get_depth(channel_in * 6), se_ratio, prefix)
    
    h = layers.Conv2D(filters, kernel_size=1, padding="same", use_bias=False, name=prefix+"project")(h)
    h = layers.BatchNormalization(fepsilon=1e-3, momentum=0.999, name=prefix + 'project/BatchNorm')(h)
    if stride == 1 and channel_in == filters:
        h = layers.Add(name=prefix + "Add")([shortcut, h])
    return h
        
def output_block(inputs, filters, activation, output_size, inlcude_top):
    h = layers.Conv2D(filters, 1, padding="same", use_bias=False, name="Conv_1")(h)
    h = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name="Conv_1/BatchNorm")(h)
    h = activation(h)
    h = layers.Conv2D(filters, 1, padding="same", use_bias=True, name="Conv_2")(h)
    h = activation(h)
    
    if inlcude_top:
        h = layers.GlobalAveragePooling2D()(h)
        h = layers.Reshape((1, 1, filters))(h)
        h = layers.Dropout(0.2)(h)
        h = layers.Conv2D(output_size, 1, padding="same", name="logits")(h)
        h = layers.Flatten()(h)
        h = layers.Softmax()(h)
    else:
        h = layers.GlobalAveragePooling2D(name="avg_pool")(h)
    return h

def _mobilenet(inputs, stack_fn, minimalistic, output_size, include_top):
    if minimalistic:
        kernel = 3
        activation = K.relu
        se_ratio = None
    else:
        kernel = 5
        activation = hard_swish
    h = input_block(inputs, )
        

def MobilenetV3Small(output_size=1000, minimalistic=False, include_top=True, name="MobileNetV3-Small"):
    ...
        
    
    
def MobilenetV3Large(output_size=1000, include_top=True):
    ...
        


if __name__ == "__main__":
    model2 = applications.MobileNetV3Small(weights=None)
    model2.summary()