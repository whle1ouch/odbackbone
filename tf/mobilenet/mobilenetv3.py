from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import applications


class H_Swish(layers.Layer):
    
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
    
    def call(self, inputs):
        return inputs * K.sigmoid(inputs)
    
    def get_config(self):
        return super().get_config()
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    
def mobilenet_block_v3(inputs, filters, kernel_size, stride, se_ratio, activation, block_id):
    shortcut = inputs
    prefix = 'expanded_conv/'
    channel_in = K.int_shape(inputs)[-1]
    if block_id:
        prefix = 'expanded_conv_{}/'.format(block_id)
 
    

if __name__ == "__main__":
    model2 = applications.MobileNetV3Small(weights=None)
    model2.summary()