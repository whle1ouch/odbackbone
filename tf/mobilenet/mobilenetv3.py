from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

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
    h = layers.GlobalAveragePooling2D(keepdims=True, 
                                      name=prefix+"squeeze_exite/AvgPool"
                                      )(inputs)
    h = layers.Conv2D(get_depth(filters*se_ratio), 1, 
                      strides=1, 
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
    
def mobilenet_block_v3(inputs, expansion, filters, kernel_size, stride, se_ratio, activation, block_id):
    channel_in = K.int_shape(inputs)[-1]
    shortcut = inputs
    prefix = 'expanded_conv/'
    if block_id:
        prefix = 'expanded_conv_{}/'.format(block_id)
        h = layers.Conv2D(get_depth(channel_in * expansion), 
                          1, 
                          padding="same", 
                          use_bias=False, 
                          name=prefix+"expand")(inputs)
        h = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix+"expand/BatchNorm")(h)
        h = activation(h)
    else:
        h = inputs
    
    if stride == 2:
        input_size = K.int_shape(h)[-3: -1]
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
        correct = (kernel_size//2, kernel_size//2)
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
        h = se_block(h, get_depth(channel_in * expansion), se_ratio, prefix)
    
    h = layers.Conv2D(filters, kernel_size=1, padding="same", use_bias=False, name=prefix+"project")(h)
    h = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'project/BatchNorm')(h)
    if stride == 1 and channel_in == filters:
        h = layers.Add(name=prefix + "Add")([shortcut, h])
    return h
        
def output_block(inputs, last_point_filters, activation, output_size, inlcude_top):
    last_conv_filters= get_depth(K.int_shape(inputs)[-1] * 6)
    h = layers.Conv2D(last_conv_filters, 1, padding="same", use_bias=False, name="Conv_1")(inputs)
    h = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name="Conv_1/BatchNorm")(h)
    h = activation(h)
    h = layers.GlobalAveragePooling2D(keepdims=True)(h)
    h = layers.Conv2D(last_point_filters, 1, padding="same", use_bias=True, name="Conv_2")(h)
    h = activation(h)
    
    if inlcude_top:
        h = layers.Dropout(0.2)(h)
        h = layers.Conv2D(output_size, 1, padding="same", name="logits")(h)
        h = layers.Flatten()(h)
        h = layers.Softmax()(h)
    else:
        h = layers.GlobalAveragePooling2D(name="avg_pool")(h)
    return h

def _mobilenet(stack_fn, last_point_filters, minimalistic, output_size, include_top, name):
    if minimalistic:
        kernel = 3
        activation = K.relu
        se_ratio = None
    else:
        kernel = 5
        activation = hard_swish
        se_ratio = 0.25
    inputs = layers.Input((224, 224, 3))
    h = input_block(inputs, 16, 2, activation)
    output = stack_fn(h, kernel, se_ratio, activation)
    pred = output_block(output, last_point_filters, activation, output_size, include_top)
    model = Model(inputs, pred, name=name)
    return model
        

def MobilenetV3Small(output_size=1000, minimalistic=False, include_top=True, name="MobileNetV3-Small"):
    
    def bottleneck_fn(inputs, kernel_size, se_ratio, activation):
        h = mobilenet_block_v3(inputs, 1,  get_depth(16), 3, 2, se_ratio, K.relu, 0)
        h = mobilenet_block_v3(h, 72./16, get_depth(24), 3, 2, None, K.relu, 1)
        h = mobilenet_block_v3(h, 88./24, get_depth(24), 3, 1, None, K.relu, 2)
        h = mobilenet_block_v3(h, 4, get_depth(40), kernel_size, 2, se_ratio, activation, 3)
        h = mobilenet_block_v3(h, 6, get_depth(40), kernel_size, 1, se_ratio, activation, 4)
        h = mobilenet_block_v3(h, 6, get_depth(40), kernel_size, 1, se_ratio, activation, 5)
        h = mobilenet_block_v3(h, 3, get_depth(48), kernel_size, 1, se_ratio, activation, 6)
        h = mobilenet_block_v3(h, 3, get_depth(48), kernel_size, 1, se_ratio, activation, 7)
        h = mobilenet_block_v3(h, 6, get_depth(96), kernel_size, 2, se_ratio, activation, 8)
        h = mobilenet_block_v3(h, 6, get_depth(96), kernel_size, 1, se_ratio, activation, 9)
        h = mobilenet_block_v3(h, 6, get_depth(96), kernel_size, 1, se_ratio, activation, 10)
        return h
    return _mobilenet(bottleneck_fn, 1024, minimalistic, output_size, include_top, name)
        
        
    
    
def MobilenetV3Large(output_size=1000, minimalistic=False, include_top=True, name="MobileNetV3-Large"):
    
    def bottleneck_fn(inputs, kernel_size, se_ratio, activation):
        h = mobilenet_block_v3(inputs, 1, get_depth(16), 3, 1, None, K.relu, 0)
        h = mobilenet_block_v3(h, 4, get_depth(24), 3, 2, None, K.relu, 1)
        h = mobilenet_block_v3(h, 3, get_depth(24), 3, 1, None, K.relu, 2)
        h = mobilenet_block_v3(h, 3, get_depth(40), kernel_size, 2, se_ratio, K.relu, 3)
        h = mobilenet_block_v3(h, 3, get_depth(40), kernel_size, 1, se_ratio, K.relu, 4)
        h = mobilenet_block_v3(h, 3, get_depth(40), kernel_size, 1, se_ratio, K.relu, 5)
        h = mobilenet_block_v3(h, 6, get_depth(80), 3, 2, None, activation, 6)
        h = mobilenet_block_v3(h, 2.5, get_depth(80), 3, 1, None, activation, 7)
        h = mobilenet_block_v3(h, 2.3, get_depth(80), 3, 1, None, activation, 8)
        h = mobilenet_block_v3(h, 2.3, get_depth(80), 3, 1, None, activation, 9)
        h = mobilenet_block_v3(h, 6, get_depth(112), 3, 1, se_ratio, activation, 10)
        h = mobilenet_block_v3(h, 6, get_depth(112), 3, 1, se_ratio, activation, 11)
        h = mobilenet_block_v3(h, 6, get_depth(160), kernel_size, 2, se_ratio, activation, 12)
        h = mobilenet_block_v3(h, 6, get_depth(160), kernel_size, 1, se_ratio, activation, 13)
        h = mobilenet_block_v3(h, 6, get_depth(160), kernel_size, 1, se_ratio, activation, 14)
        return h
    return _mobilenet(bottleneck_fn, 1280, minimalistic, output_size, include_top, name)
    
        


if __name__ == "__main__":
    model = MobilenetV3Large()
    model.summary()