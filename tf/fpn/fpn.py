from tensorflow.keras import layers
from tensorflow.keras.models import Model


def fpn_block(inputs, filters, num_conv, expand=1):
    filters_ = int(filters * expand)
    h = inputs
    for _ in range(1, num_conv+1):
        h = layers.Conv2D(filters_, (3, 3), activation="relu", padding="same")(h)
    h = layers.Conv2D(filters, 3, 2, activation="relu", padding="same")(h)
    return h

def fpn(inputs):
    
    h = fpn_block(inputs, 32, 2)
    feat1 = layers.Conv2D(256, 1, 1)(h)
    h = fpn_block(h, 64, 2)
    feat2 = layers.Conv2D(256, 1, 1)(h)
    h = fpn_block(h, 128, 4)
    feat3 = layers.Conv2D(256, 1, 1)(h)
    h = fpn_block(h, 256, 4)
    feat4 = layers.Conv2D(256, 1, 1)(h)
    h = fpn_block(h, 512, 4)
    feat5 = layers.Conv2D(256, 1, 1)(h)
    return feat1, feat2, feat3, feat4, feat5

    
if __name__ == "__main__":
    inputs = layers.Input((208, 208, 3))
    preds = fpn(inputs)
    model = Model(inputs, preds)
    model.summary()