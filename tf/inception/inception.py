from tensorflow.keras import layers
from tensorflow.keras.models import Model

def input_block(inputs, filters_7, filters_3):
    h = layers.Conv2D(filters_7, 7, 2, padding="same", name="Conv1")(inputs)
    h = layers.ReLU(name="Conv1_relu")(h)
    h = layers.MaxPooling2D((3, 3), 2, padding="same", name="Conv1_pool")(h)
    
    h = layers.Conv2D(filters_3, 3, 1, padding="same", name="Conv2")(h)
    h = layers.ReLU(name="Conv2_relu")(h)
    h = layers.MaxPooling2D((3, 3), 2, padding="same", name="Conv2_pool")(h)
    return h

def inception_block(inputs, filters_1, filters_3_reduce,  filters_3, filters_5_reduce, filters_5, filters_pool, name):
    """
    inception block of googlenet, including 4 pathes of feature extraction:
    1x1 conv, 3x3 conv, 5x5 conv, and max pool, finnaly concat these result 
    by channel axis

    Args:
        inputs (tf.Tensor): input tensor
        filters_1 (int): filter nums in 1x1 conv path
        filters_3_reduce (int): filter nums of 1x1 conv in 3x3 conv path, which can squeeze the channels
        filters_3 (int): filter nums of 3x3 conv in 3x3 conv path
        filters_5_reduce (int): filter nums of 1x1 conv in 5x5 conv path, which can squeeze the channels
        filters_5 (int): filter nums of 5x5 conv in 5x5 conv path
        filters_pool (int): filter nums of 1x1 conv in max pool
        name (str): prefix of block name

    Returns:
        tf.Tensor: output
    """
    h1 = layers.Conv2D(filters_1, 1, padding="same", name=name+"_conv1")(inputs)
    h1 = layers.ReLU(name=name+"__conv1_relu")(h1)
    
    h2 = layers.Conv2D(filters_3_reduce, 1, padding="same", name=name+"_conv2_1")(inputs)
    h2 = layers.Conv2D(filters_3, 3, padding="same", name=name+"_conv2_2")(h2)
    h2 = layers.ReLU(name=name+"_conv2_relu")(h2)
    
    h3 = layers.Conv2D(filters_5_reduce, 1, padding="same", name=name+"_conv3_1")(inputs)
    h3 = layers.Conv2D(filters_5, 5, padding="same", name=name+"_conv3_2")(h3)
    h3 = layers.ReLU(name=name+"_conv3_relu")(h3)
    
    h4 = layers.MaxPooling2D((3, 3), 1, padding="same", name=name+"_pool")(inputs)
    h4 = layers.Conv2D(filters_pool, 1, padding="same")(h4)
    h4 = layers.ReLU(name=name+"_pool_relu")(h4)
    
    h = layers.Concatenate(name=name+"_concat")([h1, h2, h3, h4])
    return h

def output_block(inputs, output_size, include_top):
    h = layers.GlobalAveragePooling2D(name="global_pool")(inputs)
    if include_top:
        h = layers.Dense(output_size, name="Dense")(h)
        h = layers.Softmax(name="prediction")(h)
    return h


def Inception(output_size=1000, include_top=True, name="GoogleNet"):
    image_inputs = layers.Input((224, 224, 3))
    h = input_block(image_inputs, 64, 192)
    
    h = inception_block(h, 64, 96, 128, 16, 32, 32, "inception_3a")
    h = inception_block(h, 128, 128, 196, 32, 96, 64, "inception_3b")
    h = layers.MaxPooling2D((3, 3), 2, padding="same", name="max_pool_3")(h)
    
    h = inception_block(h, 192, 96, 208, 16, 48, 64, "inception_4a")
    h = inception_block(h, 160, 112, 224, 24, 64, 64, "inception_4b")
    h = inception_block(h, 128, 128, 256, 24, 64, 64, "inception_4c")
    h = inception_block(h, 112, 144, 288, 32, 64, 64, "inception_4d")
    h = inception_block(h, 256, 160, 320, 32, 128, 128, "inception_4e")
    h = layers.MaxPooling2D((3, 3), 2, padding="same", name="max_pool_4")(h)
    
    h = inception_block(h, 256, 160, 320, 32, 128, 128, "inception_5a")
    h = inception_block(h, 384, 192, 384, 48, 128, 128, "inception_5b")
    
    output = output_block(h, output_size, include_top)
    model = Model(image_inputs, output, name=name)
    return model
    
    
    
if __name__ == "__main__":
    model = Inception()
    model.summary()    
    
