from tensorflow.keras import layers
from tensorflow.keras.models import Model


def conv2d_bn(inputs, filters, kernel, strides=(1, 1), padding="same"):
    h = layers.Conv2D(filters, kernel, strides=strides, padding=padding, use_bias=False)(inputs)
    h = layers.BatchNormalization(epsilon=1.001e-5, scale=False, )(h)
    h = layers.ReLU()(h)
    return h

def inception_stem(inputs):
    
    h = conv2d_bn(inputs, 32, 3, 2, padding="valid") # (299 - 2) / 2= 149
    h = conv2d_bn(h, 32, 3, padding="valid") #149 - 2 = 147
    h = conv2d_bn(h, 64, 3) 
    
    h1 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="valid")(h)
    h2 = conv2d_bn(h, 96, 3, 2, padding="valid")
    h = layers.Concatenate()([h1, h2]) # (147-2)/2 = 73
    
    h1 = conv2d_bn(h, 64, 1)
    h1 = conv2d_bn(h1, 96, 3, padding='valid') 
    
    h2 = conv2d_bn(h, 64, 1)
    h2 = conv2d_bn(h2, 64, (1, 7))
    h2 = conv2d_bn(h2, 64, (7, 1))
    h2 = conv2d_bn(h2, 96, 3, padding='valid') 
    
    h = layers.Concatenate()([h1, h2]) # 73 - 2= 71
    
    h1 = conv2d_bn(h, 192, 3, 2, padding="valid")
    h2 = layers.MaxPooling2D((3, 3), 2)(h)
    h = layers.Concatenate()([h1, h2]) # (71-2) / 2 = 35
    return h

def inception_A(inputs):
    h1 = conv2d_bn(inputs, 96, 1)

    h2 = conv2d_bn(inputs, 64, 1)
    h2 = conv2d_bn(h2, 96, 3)

    h3 = conv2d_bn(inputs, 64, 1)
    h3 = conv2d_bn(h3, 96, 3)
    h3 = conv2d_bn(h3, 96, 3)

    h4 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    h4 = conv2d_bn(h4, 96, 1)

    h = layers.Concatenate()([h1, h2, h3, h4])
    return h


def inception_B(inputs):
    h1 = conv2d_bn(inputs, 384, 1, 1)

    h2 = conv2d_bn(inputs, 192, 1)
    h2 = conv2d_bn(h2, 224, (1, 7))
    h2 = conv2d_bn(h2, 256, (7, 1))

    h3 = conv2d_bn(inputs, 192, 1)
    h3 = conv2d_bn(h3, 192, (7, 1))
    h3 = conv2d_bn(h3, 224, (1, 7))
    h3 = conv2d_bn(h3, 224, (7, 1))
    h3 = conv2d_bn(h3, 256, (1, 7))

    h4 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    h4 = conv2d_bn(h4, 128, 1)

    h = layers.Concatenate()([h1, h2, h3, h4])
    return h


def inception_C(inputs):

    h1 = conv2d_bn(inputs, 256, 1)

    h2 = conv2d_bn(inputs, 384, 1)
    h21 = conv2d_bn(h2, 256, (1, 3))
    h22 = conv2d_bn(h2, 256, (3, 1))
    h2 = layers.Concatenate()([h21, h22])

    h3 = conv2d_bn(inputs, 384, 1)
    h3 = conv2d_bn(h3, 448, (3, 1))
    h3 = conv2d_bn(h3, 512, (1, 3))
    h31 = conv2d_bn(h3, 256, (1, 3))
    h32 = conv2d_bn(h3, 256, (3, 1))
    h3 = layers.Concatenate()([h31, h32])

    h4 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    h4 = conv2d_bn(h4, 256, 1)

    h = layers.Concatenate()([h1, h2, h3, h4])
    return h


def reduction_A(inputs):
    h1 = conv2d_bn(inputs, 384, 3, 2, padding='valid')

    h2 = conv2d_bn(inputs, 192, 1)
    h2 = conv2d_bn(h2, 224, 3)
    h2 = conv2d_bn(h2, 256, 3, 2, padding='valid')

    h3 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(inputs)

    h = layers.Concatenate()([h1, h2, h3])
    return h


def reduction_B(inputs):
    h1 = conv2d_bn(inputs, 192, 1, 1)
    h1 = conv2d_bn(h1, 192, 3, 2, padding='valid')

    h2 = conv2d_bn(inputs, 256, 1)
    h2 = conv2d_bn(h2, 256, (1, 7))
    h2 = conv2d_bn(h2, 320, (7, 1))
    h2 = conv2d_bn(h2, 320, 3, 2, padding='valid')

    h3 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(inputs)

    h = layers.Concatenate()([h1, h2, h3])
    return h

def output_block(inputs, output_size, include_top):
    h = layers.GlobalAveragePooling2D()(inputs)
    if include_top:
        h = layers.Dropout(0.2)(h)
        h = layers.Dense(output_size)(h)
        h = layers.Softmax()(h)
    return h 

def InceptionV4(output_size=1000, include_top=True, name="InceptionV4"):
    image_inputs = layers.Input((299, 299, 3))
    
    h = inception_stem(image_inputs)
    
    for _ in range(4):
        h = inception_A(h)
    
    h = reduction_A(h)
    
    for _ in range(7):
        h = inception_B(h)
        
    h = reduction_B(h)
    
    for _ in range(3):
        h = inception_C(h)
        
    pred = output_block(h, output_size, include_top)
    model = Model(image_inputs, pred, name=name)
    return model

if __name__ == "__main__":
    model = InceptionV4()
    model.summary()

    