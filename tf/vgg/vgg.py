from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import activations



def vgg_block(inputs, filters, conv_num, block_id):
    prefix = f"block_{block_id}"
    h = inputs
    for i in range(1, conv_num+1):
        h = layers.Conv2D(filters, (3, 3), activation="relu", padding="same", name=prefix+"_conv"+str(i))(h)
    h = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name=prefix+"_pool")(h)
    return h
    
    
def output_block(inputs, output_size):
    h = layers.Flatten()(inputs)
    h = layers.Dense(4096, name="FC1_4096")(h)
    h = layers.Dense(4096, name="FC2_4096")(h)
    h = layers.Dense(output_size, name="FC3_1000")(h)
    h = layers.Softmax()(h)
    return h


def VGG16(output_size=1000, include_top=True, name="VGG16"):
    inputs = layers.Input((224, 224, 3))
    outputs = vgg_block(inputs, 64, 2, 1)
    outputs = vgg_block(outputs, 128, 2, 2)
    outputs = vgg_block(outputs, 256, 3, 3)
    outputs = vgg_block(outputs, 512, 3, 4)
    outputs = vgg_block(outputs, 512, 3, 5)
    if include_top:
        outputs = output_block(outputs, output_size)
    model = Model(inputs, outputs, name=name)
    return model

def VGG19(output_size=1000, include_top=True, name="VGG19"):
    inputs = layers.Input((224, 224, 3))
    outputs = vgg_block(inputs, 64, 2, 1)
    outputs = vgg_block(outputs, 128, 2, 2)
    outputs = vgg_block(outputs, 256, 4, 3)
    outputs = vgg_block(outputs, 512, 4, 4)
    outputs = vgg_block(outputs, 512, 4, 5)
    if include_top:
        outputs = output_block(outputs, output_size)
    model = Model(inputs, outputs, name=name)
    return model

if __name__ == "__main__":
    model = VGG16()
    model.summary()