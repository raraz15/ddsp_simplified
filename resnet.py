import tensorflow as tf
import tensorflow.keras.layers as tfkl
#from tensorflow.keras.models import Sequential

from utilities import ensure_4d


class NormReluConv(tf.keras.Sequential):
  """Norm -> ReLU -> Conv layer."""

  def __init__(self, ch, k, s, **kwargs):
    """Downsample frequency by stride."""
    layers = [
        tfkl.LayerNormalization(),
        tfkl.Activation(tf.nn.relu),
        tfkl.Conv2D(ch, (k, k), (1, s), padding='same')
    ]
    super().__init__(layers, **kwargs)


class ResidualLayer(tfkl.Layer):
    """A single layer for ResNet, with a bottleneck."""

    def __init__(self, ch, stride=1, shortcut=True):
        super().__init__(name='ResLayer')
        ch_out = 4 * ch
        self.shortcut = shortcut

        self.norm_input = tfkl.LayerNormalization()

        if self.shortcut:
            self.conv_proj = tfkl.Conv2D(ch_out, (1, 1), (1, stride), padding='same', name='conv_proj')
        layers = [
            tfkl.Conv2D(ch, (1, 1), (1, 1), padding='same'),
            NormReluConv(ch, 3, stride),
            NormReluConv(ch_out, 1, 1),
        ]
        self.bottleneck = tf.keras.Sequential(layers, name='bottleneck')        
             
    def call(self, inputs):
        x = inputs
        r = x

        x = ensure_4d(x)
        x = tf.nn.relu(self.norm_input(x))

        # The projection shortcut should come after the first norm and ReLU
        # since it performs a 1x1 convolution.
        r = self.conv_proj(x) if self.shortcut else r
        x = self.bottleneck(x)
        return x + r
        
        
class ResidualStack(tfkl.Layer):
    """LayerNorm -> ReLU -> Conv layer."""

    def __init__(self,
               filters,
               block_sizes,
               strides,
               **kwargs):
        super().__init__(**kwargs)
        layers = []
        for (ch, n_layers, stride) in zip(filters, block_sizes, strides):
            # Only the first block per residual_stack uses shortcut and strides.
            layers.append(ResidualLayer(ch, stride, shortcut=True))
            # Add the additional (n_layers - 1) layers to the stack.
            for _ in range(1, n_layers):
                layers.append(ResidualLayer(ch, 1, shortcut=False))
                              
        layers.append(tfkl.LayerNormalization())
        layers.append(tfkl.Activation("relu"))
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    
class ResNet(tfkl.Layer):

    def __init__(self, ch=32, blocks=[3,4,6], **kwargs): #, use_norm=True
        super().__init__(**kwargs)
        self.layers = [
            tfkl.Conv2D(64, (7, 7), (1, 2), padding='same'),
            tfkl.MaxPool2D(pool_size=(1, 3), strides=(1, 2), padding='same'),
            ResidualStack([ch, 2*ch, 4*ch], blocks, [1, 2, 2]), #, use_norm
            ResidualStack([8 * ch], [3], [2]) #, use_norm
        ]

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x