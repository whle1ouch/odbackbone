from cv2 import groupRectangles
from pip import main
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from functools import reduce, wraps

def compose_method(f1, f2):
    def wrapper(*args, **kwargs):
        return f2(f1(*args, **kwargs))
    
    return wrapper

def compose(*funcs):
    return reduce(compose_method, funcs)

@wraps(layers.Conv2D)
def ConvBnRelu2D(filters, kernel_size, strides=(1, 1), padding="same", use_bias=False, name=None, **kwargs):
    prefix = "Conv" if name is None else name
    return compose(layers.Conv2D(filters, kernel_size, strides, padding=padding, use_bias=use_bias, name=prefix+"_conv", **kwargs),
                   layers.BatchNormalization(name=prefix+"_bn"),
                   layers.ReLU(name=prefix+"_relu"))
    

def resnext_block_v1(inputs, C, filters_d, filters_out, strides=1, name=None):
    prefix = "res" if name is None else name
    hs = []
    for i in range(1, C+1):
        name = prefix + "_c" + str(i)
        h = ConvBnRelu2D(filters_d, 1, strides=strides, name=name+"_conv1")(inputs)
        h = ConvBnRelu2D(filters_d, 3, name=name+"_conv2")(h)
        h = ConvBnRelu2D(filters_out, 1, name=name+"_conv3")(h)
        hs.append(h)
    h = layers.Add(name=prefix+"_merge")(hs)
    if strides == 1 and K.int_shape(inputs)[-1] == filters_out:
        shortcut = inputs
    else:
        shortcut = ConvBnRelu2D(filters_out, 1, strides, name=prefix+"_shortcut")(inputs)
    h = layers.Add(name=prefix+"_add")([h, shortcut])
    return h

def resnext_block_v2(inputs, C, filters_d, filters_out, strides=1, name=None):
    prefix = "res" if name is None else name
    hs = []
    for i in range(1, C+1):
        name = prefix + "_c" + str(i)
        h = ConvBnRelu2D(filters_d, 1, strides=strides, name=name+"_conv1")(inputs)
        h = ConvBnRelu2D(filters_d, 3, name=name+"_conv2")(h)
        hs.append(h)
    h = layers.Concatenate(name=prefix+"_merge")(hs)
    h = ConvBnRelu2D(filters_out, 1, name=name+"_conv")(h)
    if strides == 1 and K.int_shape(inputs)[-1] == filters_out:
        shortcut = inputs
    else:
        shortcut = ConvBnRelu2D(filters_out, 1, strides, name=prefix+"_shortcut")(inputs)
    h = layers.Add(name=prefix+"_add")([h, shortcut])
    return h

def resnext_block_v3(inputs, C, filters_d, filters_out, strides=1, name=None):
    prefix = "res" if name is None else name
    pre = ConvBnRelu2D(filters_d * C, 1, strides=strides, name=name+"_conv1")(inputs)
    h = ConvBnRelu2D(filters_d * C, 3, name=name+"_group_conv", groups=C)(pre)
    h = ConvBnRelu2D(filters_out, 1, name=name+"_conv2")(h)
    if strides == 1 and K.int_shape(inputs)[-1] == filters_out:
        shortcut = inputs
    else:
        shortcut = ConvBnRelu2D(filters_out, 1, strides, name=prefix+"_shortcut")(inputs)
    h = layers.Add(name=prefix+"_add")([h, shortcut])
    return h


BLOCK_TYPE = {1: resnext_block_v1, 2: resnext_block_v2, 3: resnext_block_v3}

def make_resnext_block(inputs, num_blocks, C, filters_conv, filters_out, strides, name, block_type=1):
    if block_type not in (1, 2, 3):
        raise ValueError(f"no such this ResNeXt block of id {block_type}")
    resnext_block = BLOCK_TYPE[block_type]
    filters_d = filters_conv // C
    h = resnext_block(inputs, C, filters_d, filters_out, strides, name=name+"_1")
    for i in range(2, num_blocks+1):
        h = resnext_block(h, C, filters_d, filters_out, name=name+"_"+str(i))
    return h

def input_stem(inputs):
    h = ConvBnRelu2D(64, 7, 2, name="stem_conv1")(inputs)
    h = layers.MaxPooling2D(3, 2, padding="same")(h)
    return h

def output_block(inputs, output_size, include_top):
    h = layers.GlobalAveragePooling2D()(inputs)
    if include_top:
        h = layers.Dense(output_size, name="logit")(h)
        h = layers.Softmax(name="prediction")(h)
    return h

    
def ResNeXt50(output_size=1000, include_top=True, name="ResNeXt50"):
    image_input = layers.Input((224, 224, 3))
    h = input_stem(image_input)
    
    h = make_resnext_block(h, 3, 32, 128, 256, 1, name="conv2")
    
    h = make_resnext_block(h, 4, 32, 256, 512, 2, name="conv3")
    
    h = make_resnext_block(h, 6, 32, 512, 1024, 2, name="conv4")
    
    h = make_resnext_block(h, 3, 32, 1024, 2048, 2, name="conv5")

    
    pred = output_block(h, output_size, include_top)
    
    model = Model(image_input, pred, name=name)
    return model
        
        
    

if __name__ == "__main__":
   model = ResNeXt50()
   model.summary()