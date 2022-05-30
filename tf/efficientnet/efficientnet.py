from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import math

DEFAULT_BLOCKS_ARGS = [{
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 32,
    'filters_out': 16,
    'expand_ratio': 1,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 2,
    'filters_in': 16,
    'filters_out': 24,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 2,
    'filters_in': 24,
    'filters_out': 40,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 3,
    'filters_in': 40,
    'filters_out': 80,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 3,
    'filters_in': 80,
    'filters_out': 112,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 4,
    'filters_in': 112,
    'filters_out': 192,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 192,
    'filters_out': 320,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}]

def correct_pad(inputs, kernel_size):
    input_size = K.int_shape(inputs)[1: 3]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return ((correct[0] - adjust[0], correct[0]), 
            (correct[1] - adjust[1], correct[1]))
    
def round_filters(filters, divisor):
    new_filters = max(divisor, int(filters + divisor /2) // divisor * divisor)
    if new_filters < 0.9 * divisor:
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, depth_coefficient):
    return int(math.ceil(depth_coefficient * repeats))

def block(inputs,
          activation='swish',
          drop_rate=0.,
          name='',
          filters_in=32,
          filters_out=16,
          kernel_size=3,
          strides=1,
          expand_ratio=1,
          se_ratio=0.,
          id_skip=True):
    ...
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        h = layers.Conv2D(filters, 1, padding="same", use_bias=False, name=name+"expand_conv")(inputs)
        h = layers.BatchNormalization(name=name+"expand_bn")(h)
        h = layers.Activation(activation, name=name+"expand_ac")(h)
    else:
        h = inputs
    if strides == 2:
        h = layers.ZeroPadding2D(correct_pad(h, kernel_size), name=name+"dwconv_pad")(h)
        conv_pad = "valid"
    else:
        conv_pad = "same"
    h = layers.DepthwiseConv2D(kernel_size, strides=strides, padding=conv_pad, use_bias=False, name=name+"dwconv")(h)
    h = layers.BatchNormalization(name=name+"bn")(h)
    h = layers.Activation(activation, name=name+"ac")(h)
    
    if 0 < se_ratio <=1 :
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling2D(keepdims=True, name=name+"se_squeeze")(h) 
        se = layers.Conv2D(filters_se, 1, padding="same", activation=activation, 
                           name=name+"se_reduce")(se)
        se = layers.Conv2D(filters, 1, padding="same", activation="sigmoid", name=name+"se_expand")(se)
        h = layers.Multiply(name=name+"se_excite")([h, se])
    
    h = layers.Conv2D(filters_out, 1, padding="same", use_bias=False, name=name+"project_conv")(h)
    h = layers.BatchNormalization(name=name+"project_bn")(h)
    if id_skip and strides == 1 and filters_in == filters_out:
        if drop_rate > 0:
            h = layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1), name=name+"drop")(h)
        h = layers.Add(name=name+"add")([h, inputs])
    return h
    
    
def input_block(inputs, width_coefficient, activation, depth_divisor=8):
    h = layers.ZeroPadding2D(padding=correct_pad(inputs, 3), name="stem_conv_pad")(inputs)
    h = layers.Conv2D(round_filters(32 * width_coefficient, depth_divisor),
                      3, strides=2, padding="valid", 
                      use_bias=False,
                      name="stem_conv")(h)
    h = layers.BatchNormalization(name="stem_bn")(h)
    h = layers.Activation(activation, name="steam_ac")(h)
    return h

def efficient_block(inputs, width_coefficient, depth_coefficient, blocks_args, 
                    activation, drop_connect_rate=0.2, depth_divisor=8):
    h = inputs
    b = 0
    blocks = float(sum(round_repeats(args['repeats'], depth_coefficient) for args in blocks_args))
    for (i, args) in enumerate(blocks_args):
        args["filters_in"] = round_filters(args["filters_in"] * width_coefficient, depth_divisor)
        args["filters_out"] = round_filters(args["filters_out"] * width_coefficient, depth_divisor)
        for j in range(round_repeats(args.pop("repeats"), depth_coefficient)):
            if j > 0:
                args["strides"] = 1
                args["filters_in"] = args["filters_out"]
            h = block(h, activation, drop_connect_rate * b / blocks, 
                      name='block{}{}_'.format(i + 1, chr(j + 97)), 
                      **args)
            b+=1
        
    return h

def outputs_block(inputs, width_coefficient, output_size, include_top, activation, drop_rate, depth_divisor=8):
    h = layers.Conv2D(round_filters(1280 * width_coefficient, depth_divisor), 1, padding="same", 
                      use_bias=False, name="top_conv")(inputs)
    h = layers.BatchNormalization(name="top_bn")(h)
    h = layers.Activation(activation, name="top_ac")(h)
    h = layers.GlobalAveragePooling2D(name="avg_pool")(h)
    if include_top:
        if drop_rate > 0:
            h = layers.Dropout(drop_rate, name="top_dropout")(h)
        h = layers.Dense(output_size, name="top_logit")(h)
        h = layers.Softmax(name="prediction")(h)
    return h

def _efficient_net(width_coefficient, depth_coefficient, input_size, dropout_rate,
                   include_top, output_size, name, blocks_args=DEFAULT_BLOCKS_ARGS, 
                   activation="swish", depth_divisor=8):
    inputs = layers.Input((input_size, input_size, 3))
    h = input_block(inputs, width_coefficient, activation, depth_divisor)
    h = efficient_block(h, width_coefficient, depth_coefficient, blocks_args, 
                    activation, drop_connect_rate=dropout_rate, depth_divisor=depth_divisor)
    pred = outputs_block(h, width_coefficient, output_size, include_top, activation, dropout_rate, depth_divisor)
    model = Model(inputs, pred, name=name)
    return model

def EfficientnetB0(include_top=True, output_size=1000, name="EfficientNetB0"):
    return _efficient_net(1.0, 1.0, 224, 0.2, include_top, output_size, name)

def EfficientnetB1(include_top=True, output_size=1000, name="EfficientNetB1"):
    return _efficient_net(1.0, 1.1, 240, 0.2, include_top, output_size, name)

def EfficientnetB2(include_top=True, output_size=1000, name="EfficientNetB2"):
    return _efficient_net(1.1, 1.2, 260, 0.3, include_top, output_size, name)

def EfficientnetB3(include_top=True, output_size=1000, name="EfficientNetB3"):
    return _efficient_net(1.2, 1.4, 300, 0.3, include_top, output_size, name)

def EfficientnetB4(include_top=True, output_size=1000, name="EfficientNetB4"):
    return _efficient_net(1.4, 1.8, 380, 0.4, include_top, output_size, name)

def EfficientnetB5(include_top=True, output_size=1000, name="EfficientNetB5"):
    return _efficient_net(1.6, 2.2, 456, 0.4, include_top, output_size, name)

def EfficientnetB6(include_top=True, output_size=1000, name="EfficientNetB6"):
    return _efficient_net(1.8, 2.6, 528, 0.5, include_top, output_size, name)

def EfficientnetB7(include_top=True, output_size=1000, name="EfficientNetB7"):
    return _efficient_net(2.0, 3.1, 600, 0.5, include_top, output_size, name)


if __name__ == "__main__":
    model = EfficientnetB0()
    model.summary()