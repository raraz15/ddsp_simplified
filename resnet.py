import tensorflow.keras.layers as tfkl
from tensorflow.keras.models import Sequential


class ResidualLayer(tfkl.Layer):
    
    def __init__(self, k_filters, s_freq=1, shortcut=True, use_norm=True, name='ResLayer'):
        super().__init__(name=name)
        
        self.layer_norm = tfkl.LayerNormalization
        self.act = tfkl.Activation('relu')
        
        self.use_norm = use_norm
        self.shortcut = shortcut
        
        conv1 = tfkl.Conv2D(k_filters, kernel_size=1, strides=1, padding='same', activation=None)
        conv2 = tfkl.Conv2D(k_filters, kernel_size=3, strides=(1, s_freq), padding='same', activation=None)
        conv3 = tfkl.Conv2D(k_filters*4, kernel_size=1, strides=1, padding='same', activation=None)
        
        self.subblock1 = self.create_subblock(conv1)
        self.subblock2 = self.create_subblock(conv2)
        self.subblock3 = self.create_subblock(conv3)
        
        if self.shortcut:
            res_conv = tfkl.Conv2D(k_filters*4, kernel_size=1, strides=(1, s_freq), padding='same', activation=None)
            self.subblock_res = self.create_subblock(res_conv)

        
    def call(self, x):
        
        h = x
        
        x = self.subblock1(x)
        x = self.subblock2(x)
        x = self.subblock3(x)
        if self.shortcut:
            h = self.subblock_res(h)
        
        return h + x
    
    def create_subblock(self, conv):    
        if self.use_norm:
            return Sequential([self.layer_norm(), self.act, conv])
        else:
            return Sequential([self.act, conv])
        
        
class ResidualStack(tfkl.Layer):
    def __init__(self,
               filters,
               block_sizes,
               strides,
               use_norm,
               **kwargs):
        super().__init__(**kwargs)
        layers = []
        for (ch, n_layers, stride) in zip(filters, block_sizes, strides):
            
            layers.append(ResidualLayer(ch, stride, True, use_norm))
            for _ in range(1, n_layers):
                layers.append(ResidualLayer(ch, 1, False, use_norm))
                              
        layers.append(tfkl.LayerNormalization())
        layers.append(tfkl.Activation("relu"))
        self.layers = layers

    def __call__(self, inputs):
        x = inputs

        for layer in self.layers:
            x = layer(x)
        return x

    
class ResNet(tfkl.Layer):

    def __init__(self, ch=32, use_norm=True, blocks=[3,4,6], **kwargs):
        super().__init__(**kwargs)
        self.layers = [
            tfkl.Conv2D(64, (7, 7), (1, 2), padding='same'),
            tfkl.MaxPool2D(pool_size=(1, 3), strides=(1, 2), padding='same'),
            ResidualStack([ch, 2 * ch, 4 * ch], blocks, [1, 2, 2], use_norm),
            ResidualStack([8 * ch], [3], [2], use_norm)
        ]

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x