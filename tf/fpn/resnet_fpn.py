from tensorflow.keras import layers
from tensorflow.keras.models import Model


def conv_block(inputs, kernel, filters, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    
    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides)(inputs)
    shortcut = layers.BatchNormalization()(shortcut)
    
    h = layers.Conv2D(filters1, 1, strides)(inputs)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    
    h = layers.Conv2D(filters2, kernel, padding="same")(h)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    
    h = layers.Conv2D(filters3, 1)(h)
    h = layers.BatchNormalization()(h)
    
    h = layers.Add()([h, shortcut])
    h = layers.ReLU()(h)
    return h


def identity_block(inputs, kernel, filters, trainable=True, use_bias=True):
    nb_filters1, nb_filters2, nb_filters3 = filters
    
    h = layers.Conv2D(nb_filters1, 1, trainable=trainable, use_bias=use_bias, kernel_initializer='normal')(inputs)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    
    h = layers.Conv2D(nb_filters2, kernel,  padding="same", use_bias=use_bias, trainable=trainable, kernel_initializer='normal')(h)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    
    h = layers.Conv2D(nb_filters3, 1, trainable=trainable, use_bias=use_bias, kernel_initializer='normal')(h)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    
    h = layers.Add()([h, inputs])
    h = layers.ReLU()(h)
    return h
    
    
def ResNetFPN50(inputs, stage5=True):
    h = layers.ZeroPadding2D((3, 3))(inputs)
    h = layers.Conv2D(64, 7, 2)(h)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    
    f1 = layers.MaxPooling2D(3, 2, padding="same")(h)
    
    h = conv_block(h, 3, [64, 64, 256], 1)
    h = identity_block(h, 3, [64, 64, 256])
    f2 = identity_block(h, 3, [64, 64, 256])
    
    h = conv_block(h, 3, [128, 128, 512])
    h = identity_block(h, 3, [128, 128, 512])
    h = identity_block(h, 3, [128, 128, 512])
    f3 = identity_block(h, 3, [128, 128, 512])
    
    h = conv_block(h, 3, [256, 256, 1024], 1)
    h = identity_block(h, 3, [256, 256, 1024])
    h = identity_block(h, 3, [256, 256, 1024])
    h = identity_block(h, 3, [256, 256, 1024])
    h = identity_block(h, 3, [256, 256, 1024])
    f4 = identity_block(h, 3, [256, 256, 1024])
    if stage5:
        h = conv_block(h, 3, [512, 512, 2048], 1)
        h = identity_block(h, 3, [512, 512, 2048])
        f5 = identity_block(h, 3, [512, 512, 2048])
    else:
        f5 = None
    return f1, f2, f3, f4, f5


if __name__ == "__main__":
    inputs = layers.Input((600, 600, 3))
    preds = ResNetFPN50(inputs)
    model = Model(inputs, preds)
    model.summary()