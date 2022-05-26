from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

def conv2d_bn(inputs, filters, kernel, strides=(1, 1), use_bias=False, padding="same", activation="relu"):
    h = layers.Conv2D(filters, kernel, strides=strides, padding=padding, use_bias=use_bias)(inputs)
    if not use_bias:
        h = layers.BatchNormalization(epsilon=1.001e-5, scale=False)(h)
    if activation:
        h = layers.Activation(activation=activation)(h)
    return h

def inception_stem(inputs):
    
    h = conv2d_bn(inputs, 32, 3, 2, padding="valid")
    h = conv2d_bn(h, 32, 3, padding="valid") 
    h = conv2d_bn(h, 64, 3) 
    
    h = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="valid")(h)
    
    h = conv2d_bn(h, 80, 1, padding="valid")
    h = conv2d_bn(h, 192, 3, padding="valid")
    h = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="valid")(h)
    return h

def inception_A(inputs):
    h1 = conv2d_bn(inputs, 96, 1)

    h2 = conv2d_bn(inputs, 48, 1)
    h2 = conv2d_bn(h2, 64, 5)

    h3 = conv2d_bn(inputs, 64, 1)
    h3 = conv2d_bn(h3, 96, 3)
    h3 = conv2d_bn(h3, 96, 3)

    h4 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    h4 = conv2d_bn(h4, 64, 1)

    h = layers.Concatenate()([h1, h2, h3, h4])
    return h  

def _inception_resnet_connect(inputs, h, scale, activation):
    filters = K.int_shape(inputs)[-1]
    h = conv2d_bn(h, filters, 1, activation=None, use_bias=True)
    h = layers.Lambda(lambda x, scale: x[0] + x[1]*scale, 
                      arguments={"scale": scale}, 
                      output_shape=K.int_shape(inputs)[1:])([inputs, h])
    if activation:
        h = layers.Activation(activation=activation)(h)
    return h

def inception_resnet_A(inputs, scale, activation="relu"):
    h1 = conv2d_bn(inputs, 32, 1)
    
    h2 = conv2d_bn(inputs, 32, 1)
    h2 = conv2d_bn(h2, 32, 3)
    
    h3 = conv2d_bn(inputs, 32, 1)
    h3 = conv2d_bn(h3, 48, 3)
    h3 = conv2d_bn(h3, 64, 3)
    
    h = layers.Concatenate()([h1, h2, h3])
    h = _inception_resnet_connect(inputs, h, scale, activation)
    return h
    
    
def inception_resnet_B(inputs, scale, activation="relu"):
    h1 = conv2d_bn(inputs, 192, 1)
    
    h2 = conv2d_bn(inputs, 128, 1)
    h2 = conv2d_bn(h2, 160, (1, 7))
    h2 = conv2d_bn(h2, 192, (7, 1))
    
    h = layers.Concatenate()([h1, h2])
    h = _inception_resnet_connect(inputs, h, scale, activation)
    return h

def inception_resnet_C(inputs, scale, activation="relu"):
    h1 = conv2d_bn(inputs, 192, 1)
    
    h2 = conv2d_bn(inputs, 192, 1)
    h2 = conv2d_bn(h2, 224, (1, 3))
    h2 = conv2d_bn(h2, 256, (3, 1))
    
    h = layers.Concatenate()([h1, h2])
    h = _inception_resnet_connect(inputs, h, scale, activation)
    return h

def reduction_A(inputs):
    h1 = conv2d_bn(inputs, 384, 3, 2, padding='valid')

    h2 = conv2d_bn(inputs, 256, 1)
    h2 = conv2d_bn(h2, 256, 3)
    h2 = conv2d_bn(h2, 384, 3, 2, padding='valid')

    h3 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(inputs)

    h = layers.Concatenate()([h1, h2, h3])
    return h


def reduction_B(inputs):
    h1 = conv2d_bn(inputs, 256, 1)
    h1 = conv2d_bn(h1, 384, 3, 2, padding='valid')

    h2 = conv2d_bn(inputs, 256, 1)
    h2 = conv2d_bn(h2, 288, 3, 2, padding='valid')
    
    h3 = conv2d_bn(inputs, 256, 1)
    h3 = conv2d_bn(h3, 288, 3)
    h3 = conv2d_bn(h3, 320, 3, 2, padding="valid")
    
    h4 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(inputs)

    h = layers.Concatenate()([h1, h2, h3, h4])
    return h


def output_block(inputs, output_size, include_top):
    h = conv2d_bn(inputs, 1536, 1)
    h = layers.GlobalAveragePooling2D()(h)
    if include_top:
        h = layers.Dense(output_size)(h)
        h = layers.Softmax()(h)
    return h

def InceptionResnetV2(output_size=1000, include_top=True, name="Inception_ResNet_V2"):
    image_inputs = layers.Input((299, 299, 3))
    h = inception_stem(image_inputs)
    h = inception_A(h)
    
    # 10 times block A
    for _ in range(10):
        h = inception_resnet_A(h, 0.17)
    
    h = reduction_A(h)
    # 20 times block A
    for _ in range(20):
        h = inception_resnet_B(h, 0.1)
    
    h = reduction_B(h)
    # 10 times block B
    for _ in range(9):
        h = inception_resnet_C(h, 0.2)
    h = inception_resnet_C(h, 1, activation=None)
    
    pred = output_block(h, output_size, include_top)
    model = Model(image_inputs, pred, name=name)
    return model
    
if __name__ == "__main__":
    model = InceptionResnetV2()
    model.summary()