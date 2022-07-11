from re import X
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from functools import reduce


def compose(*funcs):
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, *kwargs)), funcs)


def actbn(layer):
     return compose(
         layer,
         layers.Activation('relu'),
         layers.BatchNormalization()
     )

def residual(layer):
    
    def wrapper(x):
        y = layer(x)
        return layers.Add()([x, y])
    
    return wrapper

def conv_mixer(filters, depth, kernel_size=9, patch_size=7, include_top=True, n_classes=1000, name="ConvMixer"):
    inputs = layers.Input((224, 224, 3))
    
    x = layers.Conv2D(filters, patch_size, patch_size)(inputs)
    
    for _ in range(depth):
        x = residual(actbn(layers.Conv2D(filters, kernel_size, 1, padding="same", groups=filters)))(x)
        # x = residual(actbn(layers.DepthwiseConv2D(kernel_size, padding="same")))(x)
        x = actbn(layers.Conv2D(filters, 1))(x)
    
    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(n_classes)(x)
        
    model = Model(inputs, x, name=name)
    return model
    

if __name__ == "__main__":
    model = conv_mixer(512, 1)
    model.summary()