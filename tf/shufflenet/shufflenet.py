from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


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

def shuffle_block_v1(inputs, mid_filters, filters, kernel_size, groups, first_group, strides):
    in_filters = K.int_shape(inputs)[-1]
    assert in_filters < filters 
    "mid_filters must be less than filters"
    if strides == 2 or strides == (2,  2):
        shortcut = layers.AveragePooling2D(3, 2, padding="same")(inputs)
        out_filters = filters - in_filters 
    else:
        shortcut = inputs
        out_filters = filters
    h = layers.Conv2D(mid_filters, 1, 1, groups=1 if first_group else groups, use_bias=False)(inputs)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    h = layers.Conv2D(mid_filters, kernel_size, strides, groups=mid_filters, padding="same", use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    if groups > 1:
        h = shuffle_channels(h, groups)
    h = layers.Conv2D(out_filters, 1, 1, groups=groups, use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    if strides == 2 or strides == (2, 2):
        h = layers.ReLU()(h)
        h = layers.Concatenate()([shortcut, h])
    else:
        h = layers.Add()([shortcut, h])
        h = layers.ReLU()(h)
    return h

def ShuffleNet(group=3,
               model_size="2.0x",
               num_class=1000,
               include_top=True,
               stage_repeats = (4, 8, 4),
               name="shuffleNet"):
    assert len(stage_repeats) <=3 and all([isinstance(x, int) for x in stage_repeats])
    "stage_repeats must be a list of positive integers with at most 3 members"
    inputs = layers.Input((224, 224, 3))
    if group == 3:
        if model_size == '0.5x':
            stage_out_channels = [-1, 12, 120, 240, 480]
        elif model_size == '1.0x':
            stage_out_channels = [-1, 24, 240, 480, 960]
        elif model_size == '1.5x':
            stage_out_channels = [-1, 24, 360, 720, 1440]
        elif model_size == '2.0x':
            stage_out_channels = [-1, 48, 480, 960, 1920]
        else:
            raise NotImplementedError
    elif group == 8:
        if model_size == '0.5x':
            stage_out_channels = [-1, 16, 192, 384, 768]
        elif model_size == '1.0x':
            stage_out_channels = [-1, 24, 384, 768, 1536]
        elif model_size == '1.5x':
            stage_out_channels = [-1, 24, 576, 1152, 2304]
        elif model_size == '2.0x':
            stage_out_channels = [-1, 48, 768, 1536, 3072]
        else:
            raise NotImplementedError
    
    h = layers.Conv2D(24, 3, 2, padding="same", use_bias=False)(inputs)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    h = layers.MaxPooling2D(3, 2, padding="same")(h)
    for i in range(len(stage_repeats)):
        filters = stage_out_channels[i+2]
        for r in range(stage_repeats[i]):
            strides = 2 if r == 0 else 1
            first_group = i == 0 and r == 0
            h = shuffle_block_v1(h, filters // 4, filters, 3, group, first_group, strides)
    h = layers.GlobalAveragePooling2D()(h)
    if include_top:
        h = layers.Flatten()(h)
        h = layers.Dense(num_class, use_bias=False)(h)
        h = layers.Softmax()(h)
    model = Model(inputs, h, name=name)
    return model

    
    
if __name__ == "__main__":
    model = ShuffleNet()
    model.summary()