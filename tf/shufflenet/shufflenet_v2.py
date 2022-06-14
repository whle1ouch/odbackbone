from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


class HardSigmoid(layers.Layer):
    def __init__(self, *args, **kwargs):
        super(HardSigmoid, self).__init__( *args, **kwargs)
        self.supports_masking = True
        
    def call(self, inputs, mask=None):
        return K.hard_sigmoid(inputs)
    
    def get_config(self):
        return super().get_config()

    def compute_output_shape(self, input_shape):
        return input_shape


def split_channels(inputs):
    in_channels = K.int_shape(inputs)[-1]
    inp = in_channels // 2
    c_hat = layers.Lambda(lambda x: x[:, :, :, :inp])(inputs)
    c = layers.Lambda(lambda x: x[:, :, :, inp:])(inputs)
    return c_hat, c


def shuffle_channels(inputs, groups):
    height, width, in_channels = K.int_shape(inputs)[1:]
    group_channels = in_channels // groups
    h = K.reshape(inputs, (-1, height, width, groups, group_channels))
    h = K.permute_dimensions(h, (0, 1, 2, 4, 3))
    h = K.reshape(h, (-1, height, width, in_channels))
    return h


def SEblock(inputs):
    input_shape = K.int_shape(inputs)
    if len(input_shape) == 4:
        h = layers.GlobalAveragePooling2D(keepdims=True)(inputs)
        h = layers.Conv2D(input_shape[-1] // 4, 1, 1, use_bias=False)(h)
        h = layers.BatchNormalization()(h)
        h = layers.ReLU()(h)
        h = layers.Conv2D(input_shape[-1], 1, 1, use_bias=False)(h)
    else:
        h = layers.GlobalAveragePooling2D(keepdims=True)(inputs)
        h = layers.Dense(input_shape[-1] // 4, use_bias=False)(h)
        h = layers.BatchNormalization()(h)
        h = layers.ReLU()(h)
        h = layers.Dense(input_shape[-1], use_bias=False)(h)
    h = K.hard_sigmoid(h)
    h = layers.Multiply()([h, inputs])
    return h


def shuffle_block_v2(inputs, mid_filters, filters, kernel_size, strides, activation="relu", se=False):
    in_filters = K.int_shape(inputs)[-1]
    assert activation in ('relu', 'hard-sigmoid')
    "activation must be one of ('relu', 'hard-sigmoid')"
    if strides == 2 or strides == (2, 2):
        h = inputs
        shortcut = inputs
    else:
        shortcut, h = split_channels(inputs)
        in_filters = in_filters // 2
    
    assert in_filters < filters 
    "mid_filters must be less than filters"
    out_filters = filters - in_filters
    
    if activation == "relu":
        act = layers.ReLU
    else:
        act = HardSigmoid
    
    h = layers.Conv2D(mid_filters, 1, 1, use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = act()(h)
    h = layers.DepthwiseConv2D(kernel_size, strides,  padding="same", use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Conv2D(out_filters, 1, 1, use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = act()(h)
    if activation == "hard-sigmoid" and se:
        h = SEblock(h)
    
    if strides == 2 or strides == (2, 2):
        shortcut = layers.DepthwiseConv2D(kernel_size, strides, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        shortcut = layers.Conv2D(in_filters, 1, 1, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        shortcut = act()(shortcut)
    h = layers.Concatenate()([h, shortcut])
    h = shuffle_channels(h, 2)
    return h


def shuffle_xception(inputs, filters, strides, activation="relu", se=False):
    in_filters = K.int_shape(inputs)[-1]
    if strides == 2 or strides == (2, 2):
        h = inputs
        shortcut = inputs
    else:
        shortcut, h = split_channels(inputs)
        in_filters = in_filters // 2
    assert in_filters < filters 
    "in_filters must be less than filters"
    assert activation in ('relu', 'hard-sigmoid')
    "activation must be one of ('relu', 'hard-sigmoid')"
    
    if activation == "relu":
        act = layers.ReLU
    else:
        act = HardSigmoid
        
    mid_filters = filters // 2
    output_filters = filters - in_filters
        
    h = layers.DepthwiseConv2D(3, strides, padding="same", use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Conv2D(mid_filters, 1, 1, use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = act()(h)
    h = layers.DepthwiseConv2D(3, 1, padding="same", use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Conv2D(filters, 1, 1, use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = act()(h)
    h = layers.DepthwiseConv2D(3, 1, padding="same", use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Conv2D(output_filters, 1, 1, use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = act()(h)
    if activation == "hard-sigmoid" and se:
        h = SEblock(h)
    
    if strides == 2 or strides == (2, 2):
        shortcut = layers.DepthwiseConv2D(3, strides, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        shortcut = layers.Conv2D(in_filters, 1, 1, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        shortcut = act()(shortcut)
    h = layers.Concatenate()([shortcut, h])
    h = shuffle_channels(h, 2)
    return h    
    

def ShuffleNetV2(input_size=224,
                 model_size="2.0x",
                 num_class=1000,
                 include_top=True,
                 stage_repeats = (4, 8, 4),
                 name="shuffleNetV2"):
    assert len(stage_repeats) <=3 and all([isinstance(x, int) for x in stage_repeats])
    "stage_repeats must be a list of positive integers with at most 3 members"
    assert model_size in ("0.5x", "1.0x", "1.5x", "2.0x") 
    "model_size must be one of ('0.5x', '1.0x', '1.5x', '2.0x')" 
    inputs = layers.Input((input_size, input_size, 3))
    if model_size == '0.5x':
        stage_out_channels = [-1, 24, 48, 96, 192, 1024]
    elif model_size == '1.0x':
        stage_out_channels = [-1, 24, 116, 232, 464, 1024]
    elif model_size == '1.5x':
        stage_out_channels = [-1, 24, 176, 352, 704, 1024]
    elif model_size == '2.0x':
        stage_out_channels = [-1, 24, 244, 488, 976, 2048]
    else:
        raise NotImplementedError
    
    h = layers.Conv2D(stage_out_channels[1], 3, 2, padding="same", use_bias=False)(inputs)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    h = layers.MaxPooling2D(3, 2, padding="same")(h)
    for i in range(len(stage_repeats)):
        filters = stage_out_channels[i+2]
        for r in range(stage_repeats[i]):
            strides = 2 if r == 0 else 1
            h = shuffle_block_v2(h, filters // 2, filters, 3, strides)
    
    h = layers.Conv2D(stage_out_channels[-1], 1, 1, use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    pred = layers.GlobalAveragePooling2D()(h)
    if model_size == "2.0x":
        pred  = layers.Dropout(0.2)(pred)
    if include_top:
        pred = layers.Dense(num_class, use_bias=False)(pred)
        pred = layers.Softmax()(pred)
    model = Model(inputs, pred, name=name)
    return model


def ShuffleNetV2_Plus(input_size=224,
                      num_class=1000, 
                      include_top=True, 
                      architecture=(0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2),
                      model_size='large',
                      stage_repeats=(4, 4, 8, 4),
                      name="shuffleNetV2_Plus"):
    assert model_size in ('large', "medium", 'small') 
    "model_size must be one of ('large', 'medium', 'small')"
    assert len(stage_repeats) <= 4 and all([isinstance(x, int) for x in stage_repeats]) 
    "stage_repeats must be a list of positive integers with at most 4 members"
    assert len(architecture) >= sum(stage_repeats) 
    "architecture must have at least the same length as stage_repeats"
    inputs = layers.Input((input_size, input_size, 3))
    if model_size == 'large':
        stage_out_channels = [-1, 16, 68, 168, 336, 672, 1280]
    elif model_size == 'medium':
        stage_out_channels = [-1, 16, 48, 128, 256, 512, 1280]
    elif model_size == 'small':
        stage_out_channels = [-1, 16, 36, 104, 208, 416, 1280]
    else:
        raise NotImplementedError
    
    h = layers.Conv2D(stage_out_channels[1], 3, 2, padding="same", use_bias=False)(inputs)
    h = layers.BatchNormalization()(h)
    h = HardSigmoid()(h)
    
    blockId = 0
    for i in range(len(stage_repeats)):
        activation = 'hard-sigmoid' if i > 0 else 'relu'
        se = True if i > 1 else False
        for r in range(stage_repeats[i]):
            strides = 2 if r == 0 else 1
            filters = stage_out_channels[i+2]
            blockTypeId = architecture[blockId]
            if blockTypeId == 0:
                h = shuffle_block_v2(h, filters // 2, filters, 3, strides, activation, se)
            elif blockTypeId == 1:
                h = shuffle_block_v2(h, filters // 2, filters, 5, strides, activation, se)
            elif blockTypeId == 2:
                h = shuffle_block_v2(h, filters // 2, filters, 7, strides, activation, se)
            elif blockTypeId == 3:
                h = shuffle_xception(h, filters, strides, activation, se)
            else:
                raise NotImplementedError
            blockId += 1
    
    h = layers.Conv2D(stage_out_channels[-1], 1, 1, use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = HardSigmoid()(h)
    pred = layers.GlobalAveragePooling2D(keepdims=True)(h)
    if include_top:
        pred = SEblock(h)
        pred = layers.Flatten()(pred)
        pred = layers.Dense(num_class, use_bias=False)(pred)
        pred = layers.Softmax()(pred)
    model = Model(inputs, pred, name=name)
    return model
    
                 

if __name__ == "__main__":
    model = ShuffleNetV2_Plus()
    model.summary()