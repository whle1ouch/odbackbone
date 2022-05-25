from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import applications
from torch import conv2d


def conv2d_bn(inputs, filters, kernel, strides=(1, 1), padding="same", name="convbn"):
    h = layers.Conv2D(filters, kernel, strides, padding=padding, use_bias=False, name=name+"_conv")(inputs)
    h = layers.BatchNormalization(epsilon=1.001e-5, name=name+"_bn")(h)
    h = layers.ReLU(name=name+"_relu")(h)
    return h

def input_block(inputs):
    h = conv2d_bn(inputs, 32, 3, strides=(2, 2), padding='valid', name="Conv1")
    h = conv2d_bn(h, 32, 3, padding='valid', name="Conv2")
    h = conv2d_bn(h, 64, 3, name="Conv3")
    h = layers.MaxPooling2D((3, 3), strides=(2, 2))(h)

    h = conv2d_bn(h, 80, 1, padding='valid', name="Conv4")
    h = conv2d_bn(h, 192, 3, padding='valid', name="Conv5")
    h = layers.MaxPooling2D((3, 3), strides=(2, 2))(h)
    
    h1 = conv2d_bn(h, 64, 1, name="mix0_p1_conv1")
    
    h2 = conv2d_bn(h, 48, 1, name="mix0_p2_conv1")
    h2 = conv2d_bn(h2, 64, 5, name="mix0_p2_conv2")
    
    h3 = conv2d_bn(h, 64, 1, name="mix0_p3_conv1")
    h3 = conv2d_bn(h3, 96, 3, name="mix0_p3_conv2")
    h3 = conv2d_bn(h3, 96, 3, name="mix0_p3_conv3")
    
    h4 = layers.AveragePooling2D((3, 3), 1, padding="same", name="mix0_p4_pool")(h)
    h4 = conv2d_bn(h4, 32, 1, name="mix0_p4_conv1")
    
    h = layers.Concatenate(name="mix0_concat")([h1, h2, h3, h4])
    return h

def inception_block_v3_1(inputs, filters_1, filters_2_reduce,  filters_2, filters_3_reduce, filters_3, filters_4, stride, name):
    hs = []
    padding = "valid" if stride == 2 else "same"
    if filters_1 > 0:
        h1 = conv2d_bn(inputs, filters_1, 1, stride, name=name+"_p1_conv1")
        hs.append(h1)
    
    h2 = inputs
    if filters_2_reduce > 0:
        h2 = conv2d_bn(h2, filters_2_reduce, 1, 1, name=name+"_p2_conv1")
    h2 = conv2d_bn(h2, filters_2, 3, stride, padding=padding, name=name+"_p2_conv2")
    hs.append(h2)
    
    h3 = conv2d_bn(inputs, filters_3_reduce, 1, 1, name=name+"_p3_conv1")
    h3 = conv2d_bn(h3, filters_3, 3, 1, name=name+"_p3_conv2")
    h3 = conv2d_bn(h3, filters_3, 3, stride, padding=padding, name=name+"_p3_conv3")
    hs.append(h3)
    
    h4 = layers.MaxPooling2D((3, 3), stride, padding=padding, name=name+"_p4_pool")(inputs)
    if filters_4 > 0:
        h4 = conv2d_bn(h4, filters_4, 1, 1, name=name+"_p4_conv1")
    hs.append(h4)
    
    h = layers.Concatenate(name=name+"_concat")(hs)
    return h


def inception_block_v3_2(inputs, filters_1, filters_2_reduce,  filters_2, filters_3_reduce, filters_3, filters_4, stride, n, name):
    hs = []
    if stride == 1:
        h1 = conv2d_bn(inputs, filters_1, 1, stride, name=name+"_p1_conv1")
        hs.append(h1)
        
        h2 = conv2d_bn(inputs, filters_2_reduce, 1, name=name+"_p2_conv1")
        h2 = conv2d_bn(h2, filters_2_reduce, (1, n), name=name+"_p2_conv2")
        h2 = conv2d_bn(h2, filters_2, (n, 1), name=name+"_p2_conv3")
        hs.append(h2)
        
        h3 = conv2d_bn(inputs, filters_3_reduce, 1, name=name+"_p3_conv1")
        h3 = conv2d_bn(h3, filters_3_reduce, (n, 1), name=name+"_p3_conv2")
        h3 = conv2d_bn(h3, filters_3_reduce, (1, n), name=name+"_p3_conv3")
        h3 = conv2d_bn(h3, filters_3_reduce, (1, n), name=name+"_p3_conv4")
        h3 = conv2d_bn(h3, filters_3, (1, n), stride, name=name+"_p3_conv5")
        hs.append(h3)
        
        h4 = layers.MaxPooling2D((3, 3), 1, padding="same", name=name+"_p4_pool")(inputs)
        h4 = conv2d_bn(h4, filters_4, 1, name=name+"_p4_conv1")
        hs.append(h4)
    else:
        h2 = conv2d_bn(inputs, filters_2_reduce, 1, name=name+"_p1_conv1")
        h2 = conv2d_bn(h2, filters_2, 3, 2, padding="valid", name=name+"_p1_conv2")
        hs.append(h2)
        
        h3 = conv2d_bn(inputs, filters_3_reduce, 1, name=name+"_p2_conv1")
        h3 = conv2d_bn(h3, filters_3_reduce, (1, n), name=name+"_p2_conv2")
        h3 = conv2d_bn(h3, filters_3_reduce, (n, 1), name=name+"_p2_conv3")
        h3 = conv2d_bn(h3, filters_3, 3, 2, padding="valid", name=name+"_p2_conv4")
        hs.append(h3)
        
        h4 = layers.MaxPooling2D((3, 3), (2, 2), name=name+"_p3_pool")(inputs)
        hs.append(h4)
    h = layers.Concatenate(name=name+"_concat")(hs)
    return h

def inception_block_v3_3(inputs, filters_1, filters_2_reduce,  filters_2, filters_3_reduce, filters_3, filters_4, stride, name):
    hs = []
    h1 = conv2d_bn(inputs, filters_1, 1, stride, name=name+"_p1_conv1")
    hs.append(h1)
    
    h2 = conv2d_bn(inputs, filters_2_reduce, 1, name=name+"_p2_conv1")
    h21 = conv2d_bn(h2, filters_2, (1, 3), name=name+"_p2_conv2")
    h22 = conv2d_bn(h2, filters_2, (3, 1), stride, name=name+"_p2_conv3")
    h2 = layers.Concatenate(name=name+"_conv2_concat")([h21, h22])
    hs.append(h2)
    
    
    h3 = conv2d_bn(inputs, filters_3_reduce, 1, name=name+"_p3_conv1")
    h3 = conv2d_bn(h3, filters_3, 3, name=name+"_p3_conv2")
    h31 = conv2d_bn(h3, filters_3, (1, 3), stride, name=name+"_p3_conv3")
    h32 = conv2d_bn(h3, filters_3, (3, 1), stride, name=name+"_p3_conv4")
    h3 = layers.Concatenate(name=name+"_conv3_concat")([h31, h32])
    hs.append(h3)
    
    h4 = layers.MaxPooling2D((3, 3), stride, padding="same", name=name+"_p4_pool")(inputs)
    h4 = conv2d_bn(h4, filters_4, 1, 1, name=name+"_p4_conv1")
    hs.append(h4)
    h = layers.Concatenate(name=name+"_concat")(hs)
    return h

def output_block(inputs, output_size, include_top):
    h = layers.GlobalAveragePooling2D()(inputs)
    if include_top:
        h = layers.Dense(output_size, name="logit")(h)
        h = layers.Softmax(name="prediction")(h)
    return h
        
def InceptionV3(output_size=1000, include_top=True, name="InceptionV3"):
    image_inputs = layers.Input((299, 299, 3))
    outputs = input_block(image_inputs)
    # 3 times block 1
    outputs = inception_block_v3_1(outputs, 64, 48, 64, 64, 96, 64, 1, name="mix1")
    outputs = inception_block_v3_1(outputs, 64, 48, 64, 64, 96, 64, 1, name="mix2")
    outputs = inception_block_v3_1(outputs, 0, 0, 384, 64, 96, 0, 2, name="mix3")
    # 5 times block 2
    outputs = inception_block_v3_2(outputs, 192, 128, 192, 128, 192, 192, 1, 7, name="mix4")
    outputs = inception_block_v3_2(outputs, 192, 160, 192, 160, 192, 192, 1, 7, name="mix5")
    outputs = inception_block_v3_2(outputs, 192, 160, 192, 160, 192, 192, 1, 7, name="mix6")
    outputs = inception_block_v3_2(outputs, 192, 192, 192, 192, 192, 192, 1, 7, name="mix7")
    outputs = inception_block_v3_2(outputs, 0, 192, 320, 192, 192, 0, 2, 7, name="mix8")
    
    # 2 times block 3
    outputs = inception_block_v3_3(outputs, 320, 384, 384, 448, 384, 192, 1, name="mix9")
    outputs = inception_block_v3_3(outputs, 320, 384, 384, 448, 384, 192, 1, name="mix10")
    
    pred = output_block(outputs, output_size, include_top)
    model = Model(image_inputs, pred, name=name)
    return model

if __name__ == "__main__":
    model = InceptionV3()
    print(model.count_params())
    model2 = applications.InceptionV3(weights=None)
    print(model2.count_params())
    