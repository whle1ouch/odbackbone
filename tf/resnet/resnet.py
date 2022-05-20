from re import X
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import applications


def res_bottle_neck(input, filters, stride=None, name=""):
    channel_in = input.shape[-1]
    channel_out = filters * 4
    if stride is None:
        stride = 1 if channel_in == channel_out else 2
    h = layers.Conv2D(filters, 1,  strides=stride, padding="same", name=name+"_1_conv")(input)
    h = layers.BatchNormalization(epsilon=1.001e-5,  name=name+"_1_bn")(h)
    h = layers.ReLU(name=name+"_1_relu")(h)
    
    h = layers.Conv2D(filters, 3, strides=1, padding="same", name=name+"_2_conv")(h)
    h = layers.BatchNormalization(epsilon=1.001e-5, name=name+"_2_bn")(h)
    h = layers.ReLU(name=name+"_2_relu")(h)
    
    h = layers.Conv2D(channel_out, 1, strides=1, padding="same", name=name+"_3_conv")(h)
    h = layers.BatchNormalization(name=name+"_3_bn")(h)
    
    if channel_in != channel_out:
        shortcut = layers.Conv2D(channel_out, 1, strides=stride, padding="same", name=name+"_0_conv")(input)
        shortcut = layers.BatchNormalization(epsilon=1.001e-5, name=name+"_0_bn")(shortcut)
    else:
        shortcut = input
        
    h = layers.Add(name=name+"_add")([h, shortcut])
    h = layers.ReLU(name=name+"_out")(h)
    return h
       
def res_bottle_neck_v2(input, filters, stride=None, name=""):
    channel_in = input.shape[-1]
    channel_out = filters * 4
    if stride is None:
        stride = 1 if channel_in == channel_out else 2
    h = layers.BatchNormalization(epsilon=1.001e-5, name=name + "_preact_bn")(input)
    h = layers.ReLU(name=name+"_preact_relu")(h)
    
    if channel_in != channel_out:
        shortcut = layers.Conv2D(filters*4, 1, strides=stride, name=name+"_0_conv")(h)
    else:
        shortcut = layers.MaxPool2D(1, strides=stride)(h) if stride > 1 else h
    
    h = layers.Conv2D(filters, 1, strides=1, use_bias=False, name=name+"_1_conv")(h)
    h = layers.BatchNormalization(epsilon=1.001e-5, name=name + "_1_bn")(h)
    h = layers.ReLU(name=name+"_1_relu")(h)
    
    h = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name+"_2_pad")(h)
    h = layers.Conv2D(filters, 3, strides=stride, use_bias=False, name=name+"_2_conv")(h)
    h = layers.BatchNormalization(epsilon=1.001e-5, name=name+"_2_bn")(h)
    h = layers.ReLU(name=name+"_2_relu")(h)
    
    h = layers.Conv2D(filters*4, 1, strides=1, name=name+"_3_conv")(h)
    h = layers.Add(name=name+"_out")([shortcut, h])
    return h
       

def make_res_block(input, filters, num_residue, stride=2, name=""):
    output = res_bottle_neck(input, filters, stride=stride, name=name+"_block1")
    for i in range(2, num_residue+1):
        output = res_bottle_neck(output, filters, name=name+"_block"+str(i))
    return output

def make_res_block_v2(input, filters, num_residue, stride=2, name=""):
    h = res_bottle_neck_v2(input, filters, name=name+"_block1")
    for i in range(2, num_residue):
        h = res_bottle_neck_v2(h, filters, name=name+"_block"+str(i))
    h = res_bottle_neck_v2(h, filters, stride=stride, name=name+"_block"+str(num_residue))
    return h

def input_block(input, preact=False):
    # resnet_conv1: (3,3) padding -> 7*7*2 conv2d -> bn -> relu -> (1,1) padding -> 3*3*2 avg pool
    output = layers.ZeroPadding2D(3, name="conv1_pad")(input)
    output = layers.Conv2D(64, kernel_size=7, strides=2, name="conv1_conv")(output)
    if not preact:
        output = layers.BatchNormalization(epsilon=1.001e-5, name="conv1_bn")(output)
        output = layers.ReLU(name="conv1_relu")(output)
    output = layers.ZeroPadding2D(1, name="pool1_pad")(output)
    output = layers.MaxPool2D(pool_size=3, strides=2, padding="valid", name="pool1_pool")(output)
    return output

def head_block(input, output_size, preact=False):
    if preact:
        h = layers.BatchNormalization(epsilon=1.001e-5, name="conv1_bn")(input)
        h = layers.ReLU(name="conv1_relu")(h)
    else:
        h = input
    output = layers.GlobalAveragePooling2D(name="avg_pool")(h)
    logit = layers.Dense(output_size)(output)
    prediction = layers.Softmax()(logit)
    return prediction

def resnet50_body(input):
    
    output = make_res_block(input, 64, 3, init_stride=1, name="conv2")
    output = make_res_block(output, 128, 4, name="conv3")
    output = make_res_block(output, 256, 6, name="conv4")
    output = make_res_block(output, 512, 3, name="conv5")

    return output
    
def Resnet50(output_size=1000, name="resnet50"):
    image_input = layers.Input((224, 224, 3))
    output = input_block(image_input)
    output = resnet50_body(output)
    prediction = head_block(output, output_size)
    model = Model(image_input, prediction, name=name)
    return model

def resnet50v2_body(input):
    output = make_res_block_v2(input, 64, 3, name="conv2")
    output = make_res_block_v2(output, 128, 4, name="conv3")
    output = make_res_block_v2(output, 256, 6, name="conv4")
    output = make_res_block(output, 512, 3, stride=1, name="conv5")

    return output
    

def Resnet50V2(output_size=1000, name="resnet50v2"):
    image_input = layers.Input((224, 224, 3))
    output = input_block(image_input, True)
    output = resnet50v2_body(output)
    prediction = head_block(output, output_size, True)
    model = Model(image_input, prediction, name=name)
    return model
    
def resnet101_body(input):
    output = make_res_block(input, 64, 3, stride=1, name="conv2")
    output = make_res_block(output, 128, 4, name="conv3")
    output = make_res_block(output, 256, 23, name="conv4")
    output = make_res_block(output, 512, 3, name="conv5")
    return output
    
def Resnet101(output_size=1000, name="resnet101"):
    image_input = layers.Input((224, 224, 3))
    output = input_block(image_input)
    output = resnet101_body(output)
    prediction = head_block(output, output_size)
    model = Model(image_input, prediction, name=name)
    return model

def resnet152_body(input):
    output = make_res_block(input, 64, 3, init_stride=1, name="conv2")
    output = make_res_block(output, 128, 8, name="conv3")
    output = make_res_block(output, 256, 36, name="conv4")
    output = make_res_block(output, 512, 3, name="conv5")
    return output

def Resnet152(output_size=1000, name="resnet152"):
    image_input = layers.Input((224, 224, 3))
    output = input_block(image_input)
    output = resnet152_body(output)
    prediction = head_block(output, output_size)
    model = Model(image_input, prediction, name=name)
    return model
    
    


if __name__ == "__main__":
    model = Resnet50V2()
    model.summary()
    print("------------------build succuss -----------------------")
    
    