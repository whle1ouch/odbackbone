from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def focus(inputs):
    return layers.Concatenate(axis=-1)([inputs[..., ::2, ::2, :], inputs[..., 1::2, 1::2, :], inputs[..., ::2, 1::2, :], inputs[..., 1::2, ::2, :]])



if __name__ == "__main__":
    inputs = layers.Input((200, 200, 3))
    h = focus(inputs)
    model = Model(inputs, h)
    model.summary()