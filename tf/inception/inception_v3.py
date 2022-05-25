from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import applications


def conv2d_bn(inputs, filters, kernel, strides=(1, 1), padding="valid", name=""):
    h = layers.Conv2D(filters, kernel, strides, padding=padding, name=name+"_conv")(inputs)
    h = layers.BatchNormalization(epsilon=1.001e-5, name=name+"_bn")(h)
    h = layers.ReLU(name=name+"_relu")(h)
    return h

def inception_block_v2(inputs, filters_1, filters_2_reduce,  filters_2, filters_3_reduce, filters_3, filters_4, stride, name):
    hs = []
    h1 = conv2d_bn(inputs, filters_1, 1, stride, padding="same", name=name+"_1x1")
    hs.append(h1)
    
    h2 = conv2d_bn(inputs, filters_2_reduce, 1, 1, padding="same", name=name+"_3x3_1")
    h2 = conv2d_bn(h2, filters_2, 3, stride, padding="same", name=name+"_3x3_2")
    hs.append(h2)
    
    h3 = conv2d_bn(inputs, filters_3_reduce, 1, 1, padding="same", name=name+"_5x5_1")
    h3 = conv2d_bn(h3, filters_3, 3, 1, padding="same", name=name+"_5x5_2")
    h3 = conv2d_bn(h3, filters_3, 3, stride, padding="same", name=name+"_5x5_3")
    hs.append(h3)
    
    h4 = layers.MaxPooling2D((3, 3), stride, padding="same", name=name+"_pool")(inputs)
    h4 = conv2d_bn(h4, filters_4, 1, 1, padding="same", name=name+"_pool_conv")
    
    h = layers.Concatenate(name=name+"_concat")(hs)
    return h


def inception_block_v2_2(inputs, filters_1, filters_2_reduce,  filters_2, filters_3_reduce, filters_3, filters_4, stride, n, name):
    hs = []
    h1 = layers.Conv2D(filters_1, 1, stride, padding="same", activation="relu", name=name+"_conv1")(inputs)
    hs.append(h1)
    
    h2 = layers.Conv2D(filters_2_reduce, 1, padding="same", name=name+"_conv2_1")(inputs)
    h2 = layers.Conv2D(filters_2_reduce, (1, n), strides=(1, stride), padding="same", name=name+"_conv2_2")(h2)
    h2 = layers.Conv2D(filters_2, (n, 1), strides=(stride, 1), padding="same", activation="relu", name=name+"_conv2_3")(h2)
    hs.append(h2)
    
    h3 = layers.Conv2D(filters_3_reduce, 1, padding="same", name=name+"_conv3_1")(inputs)
    h3 = layers.Conv2D(filters_3_reduce, (1, n), padding="same", name=name+"_conv3_2")(h3)
    h3 = layers.Conv2D(filters_3_reduce, (n, 1), padding="same", name=name+"_conv3_3")(h3)
    h3 = layers.Conv2D(filters_3_reduce, (1, n), padding="same", name=name+"_conv3_4")(h3)
    h3 = layers.Conv2D(filters_3, (n, 1), strides=stride, padding="same", activation="relu", name=name+"_conv3_5")(h3)
    hs.append(h3)
    
    h4 = layers.MaxPooling2D((3, 3), 1, stride, padding="same", name=name+"_pool")(inputs)
    h4 = layers.Conv2D(filters_4, 1, activation="relu", padding="same")(h4)
    hs.append(h4)
 
    
    h = layers.Concatenate(name=name+"_concat")(hs)
    return h

def inception_block_v2_3(inputs, filters_1, filters_2_reduce,  filters_2, filters_3_reduce, filters_3, filters_4, stride, name):
    hs = []
    h1 = layers.Conv2D(filters_1, 1, stride, padding="same", name=name+"_conv1")(inputs)
    h1 = layers.ReLU(name=name+"__conv1_relu")(h1)
    hs.append(h1)
        
    h2 = layers.Conv2D(filters_2_reduce, 1, padding="same", name=name+"_conv2_1")(inputs)
    h21 = layers.Conv2D(filters_2_reduce, (1, 3), strides=(1, stride), padding="same", activation="relu", name=name+"_conv2_22")(h2)
    h22 = layers.Conv2D(filters_2, (3, 1), strides=(stride, 1), padding="same", activation="relu", name=name+"conv2_22")(h2)
    hs += [h21, h22]
    
    h3 = layers.Conv2D(filters_3_reduce, 1, padding="same", name=name+"_conv3_1")(inputs)
    h3 = layers.Conv2D(filters_3_reduce, 3, padding="same", name=name+"_conv3_2")(h3)
    h31 = layers.Conv2D(filters_3_reduce, (1, 3), padding="same", activation="relu", name=name+"_conv3_2")(h3)
    h32 = layers.Conv2D(filters_3, (3, 1), strides=stride, padding="same", activation="relu", name=name+"_conv3_2")(h3)
    hs += [h31, h32]
    
    h4 = layers.MaxPooling2D((3, 3), 1, stride, padding="same", name=name+"_pool")(inputs)
    h4 = layers.Conv2D(filters_4, 1, padding="same")(h4)
    hs.append(h4)
    h = layers.Concatenate(name=name+"_concat")(hs)
    return h

if __name__ == "__main__":
    model = applications.InceptionV3(weights=None)
    model.summary()