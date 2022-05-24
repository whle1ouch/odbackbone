from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import applications


def input_block(inputs, filters):
    h = layers.Conv2D(filters, 7,strides=2, padding="SAME", use_bias=False, name="Conv1")(inputs)
    h = layers.BatchNormalization(epsilon=1.0001e-5, name="Conv1_bn")(h)
    h = layers.ReLU(name="Conv1_relu")(h)
    h = layers.MaxPooling2D(3, strides=2, padding="SAME", name="Conv1_pad")(h)
    return h

def dense_layer(inputs, growth_rate, drop_rate, name):
    """
    dense layer, including two convolutions and a concatenation connection which differs from resnet.

    Args:
        inputs (tf.Tensor): input tensor
        growth_rate (int): hyper parameter of dense net, representing the output channels of feature map
        drop_rate (float): dropout rate
        name (str): layer name's prefix

    Returns:
        _type_: _description_
    """
    h = layers.BatchNormalization(epsilon=1.001e-5, name=name+"bn1")(inputs)
    h = layers.ReLU(name=name+"_relu1")(h)
    h = layers.Conv2D(4 * growth_rate, 1, use_bias=False, name=name+"_conv1")(h)
    
    h = layers.BatchNormalization(epsilon=1.001e-5, name=name+"bn2")(h)
    h = layers.ReLU(name=name+"_relu2")(h)
    h = layers.Conv2D(growth_rate, 3, use_bias=False, padding="same", name=name+"_conv2")(h)
    
    if drop_rate:
        h = layers.Dropout(drop_rate)(h)
    h = layers.Concatenate(name=name+"_concat")([inputs, h])
    return h   
    
    
def dense_block(inputs, dense_num, growth_rate, drop_rate, block_id):
    """
    the main block of dense net, the dense num in paper usually contain 4 dense 
    layer with dense residue struct

    Args:
        inputs (tf.Tensor): input tensor
        dense_num (int): dense layers num, as to 4 in paper
        growth_rate (int): hyper parameter of dense net, representing the output channels of feature map
        drop_rate (float or None): dropout rate, dropout layer will be used if Non-None value is given
        block_id (int): block id used in name's prefix

    Returns:
        _type_: _description_
    """
    prefix = f"block_{block_id}"
    h = inputs
    for i in range(1, dense_num+1):
        h = dense_layer(h, growth_rate, drop_rate, name=prefix + "_dense_"+str(i))
    return h

    
def traisition_block(inputs, reduction, name):
    """
    traisition layer is used to connect between two dense layers and reduce channels and shrink 
    feature map size

    Args:
        inputs (tf.Tensor): the input tensor
        reduction (float): reduction ratio of the feature map channel, 
        name (str): block name's prefix

    Returns:
        _type_: _description_
    """
    filters = K.int_shape(inputs)[-1]
    h = layers.BatchNormalization(epsilon=1.001e-5, name=name+"_bn")(inputs)
    h = layers.ReLU(name=name+"_relu")(h)
    h = layers.Conv2D(int(filters * reduction), 1, use_bias=False, name=name+"_conv")(h)
    h = layers.AveragePooling2D((2, 2), 2, name=name+"_pad")(h)
    return h

def output_block(inputs, output_size, include_top):
    h = layers.BatchNormalization(epsilon=1.001e-5, name="out_bn")(inputs)
    h = layers.ReLU(name="out_relu")(h)
    h = layers.GlobalAveragePooling2D(name="avg_pool")(h)
    if include_top:
        h = layers.Dense(output_size, name="dense")(h)
        h = layers.Softmax(name="predictions")(h)
    return h
        

def _densenet(block_nums, init_filters, output_size, growth_rate, dropout_rate, include_top, name):
    inputs = layers.Input((224, 224, 3))
    outputs = input_block(inputs, init_filters)
    for i in range(len(block_nums)):
        outputs = dense_block(outputs, block_nums[i], growth_rate, dropout_rate, str(i+1))
        if i < len(block_nums)-1:
            outputs = traisition_block(outputs, 0.5, name="trans"+str(i+1))
    pred = output_block(outputs, output_size, include_top)
    model = Model(inputs, pred, name=name)
    return model


def Densenet121(output_size=1000, include_top=True, name="DenseNet121"):
    blocks = [6, 12, 24, 16]
    return _densenet(blocks, 64, output_size, 32, None, include_top, name)
    
def Densenet169(output_size=1000, include_top=True, name="DenseNet169"):
    blocks = [6, 12, 32, 32]
    return _densenet(blocks, 64, output_size, 32, None, include_top, name)
    
def Densenet201(output_size=1000, include_top=True, name="DenseNet201"):
    blocks = [6, 12, 48, 32]
    return _densenet(blocks, 64, output_size, 32, None, include_top, name)
    
if __name__ == "__main__":
    model1 = Densenet121()
    print(model1.count_params())
    model2 = applications.DenseNet121(weights=None)
    print(model2.count_params())