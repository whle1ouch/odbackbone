from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import applications


def entry_flow(inputs):
    """
    the backbone of xception, including 3 residual blocks of separatable conv, each of these will half feature map scale

    Args:
        inputs (tf.Tensor): input tensor

    Returns:
        tf.Tensor: output tensor
    """
    h = layers.Conv2D(32, 3, 2, use_bias=False)(inputs)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    h = layers.Conv2D(64, 3, 1, use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    
    shortcut = layers.Conv2D(128, 1, 2, 
                             padding="same", use_bias=False)(h)
    shortcut = layers.BatchNormalization()(shortcut)
    h = layers.SeparableConv2D(128, 3, padding="same", use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    h = layers.SeparableConv2D(128, 3, padding="same", use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = layers.MaxPooling2D(3, 2, padding="same")(h)
    h = layers.Add()([shortcut, h])
    
    shortcut = layers.Conv2D(256, 1, 2,
                              padding="same", use_bias=False)(h)
    shortcut = layers.BatchNormalization()(shortcut)
    h = layers.SeparableConv2D(256, 3, padding="same", use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    h = layers.SeparableConv2D(256, 3, padding="same", use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = layers.MaxPooling2D(3, 2, padding="same")(h)
    h = layers.Add()([shortcut, h])
    
    shortcut = layers.Conv2D(728, 1, 2,
                              padding="same", use_bias=False)(h)
    shortcut = layers.BatchNormalization()(shortcut)
    h = layers.SeparableConv2D(728, 3, padding="same", use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    h = layers.SeparableConv2D(728, 3, padding="same", use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = layers.MaxPooling2D(3, 2, padding="same")(h)
    h = layers.Add()([shortcut, h])
    
    return h

def separable_block(inputs, filters, kernel):
    h = layers.ReLU()(inputs)
    h = layers.SeparableConv2D(filters, kernel, padding="same", use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    h = layers.SeparableConv2D(filters, kernel, padding="same", use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    h = layers.SeparableConv2D(filters, kernel, padding="same", use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Add()([inputs, h])
    return h
    

def middle_flow(inputs, num_block=8):
    h = inputs
    for _ in range(num_block):
        h = separable_block(h, 728, 3)
    return h

def exit_flow(inputs, include_top, output_size):
    shortcut = layers.Conv2D(1024, 1, 2, padding="same", use_bias=False)(inputs)
    shortcut = layers.BatchNormalization()(shortcut)
    h = layers.ReLU()(inputs)
    h = layers.SeparableConv2D(728, 3, padding="same", use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    h = layers.SeparableConv2D(1024, 3, padding="same", use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = layers.MaxPooling2D(3, 2, padding="same")(h)
    h = layers.Add()([shortcut, h])
    
    h = layers.SeparableConv2D(1536, 3, padding="same", use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    h = layers.SeparableConv2D(2048, 3, padding="same", use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    
    h = layers.GlobalAveragePooling2D()(h)
    if include_top:
        h = layers.Dense(output_size)(h)
        h = layers.Softmax()(h)
    return h

def Xception(output_size=1000, include_top=True, name="Xception"):
    """
    a simple implementation of xception used for imageNet dataset, the input size is (299, 299)
    """
    image_input = layers.Input((299, 299, 3))
    h = entry_flow(image_input)
    h = middle_flow(h)
    pred = exit_flow(h, include_top, output_size)
    model = Model(image_input, pred, name=name)
    return model


if __name__ == "__main__":
    model = Xception()
    model.summary()
 