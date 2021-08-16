import numpy as np

import tensorflow as tf
from tensorflow.keras import layers as tfkl
import tensorflow_addons as tfa

from dsp_utils.spectral_ops import compute_loudness
from dsp_utils.core import resample
from resnet import ResNet


## ------------------------- Supervised/ Unsupervised Encoders -------------------------------------------------

class SupervisedEncoder(tfkl.Layer):
    """loudness and F0 is read from the dataset."""
    
    def __init__(self, rnn_channels=512, z_dims=32):
        
        super().__init__(name='SupervisedEncoder')        
        self.encoder_z = Encoder_z(rnn_channels, z_dims) 
    
    def call(self, features):
        z = self.encoder_z(features)        
        return {'z': z} 

class UnsupervisedEncoder(tfkl.Layer):
    
    def __init__(self, rnn_channels, z_dims, k_filters, s_freqs, R, n_fft=2048):
        
        super().__init__(name='UnsupervisedEncoder')
        
        self.encoder_z = Encoder_z(rnn_channels, z_dims)
        self.encoder_f = Encoder_f(k_filters, s_freqs, R)
        self.encoder_l = LoudnessExtractor(n_fft)
        
        self.freq_scale = tf.convert_to_tensor(440* 2**((np.arange(0,128) -69) /12), dtype=tf.float32)
    
    def call(self, features):
        
        z = self.encoder_z(features)
        f0_scores = self.encoder_f(features)
        f0 = tf.math.reduce_sum(f0_scores * self.freq_scale,axis=-1)
        l = self.encoder_l(features)
        
        return {'z': z,
                'f0_hz': f0,
                'l': l}


## ----------------------------------- Individual Encoders ---------------------------------------------


class Encoder_z(tfkl.Layer):

    def __init__(self, rnn_channels=512, z_dims=32):       
        super().__init__(name='z_encoder')

        self.norm_in = tfa.layers.InstanceNormalization(axis=-1, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")      
        self.rnn = tfkl.GRU(rnn_channels, return_sequences=True)
        self.dense_z = tfkl.Dense(z_dims)

    def call(self, features):
        mfcc = features['mfcc'] 
        z = self.norm_in(mfcc[:, :, tf.newaxis, :])[:, :, 0, :]
        z = self.rnn(z)
        z = self.dense_z(z)
        return z

class LoudnessExtractor(tfkl.Layer):
    
    def __init__(self, n_fft=2048):
        super().__init__(name='loudness_extractor')
        self.n_fft = n_fft
        
    def call(self, features):    
        return compute_loudness(features['audio'], n_fft=self.n_fft, use_tf=True)

class Encoder_f(tfkl.Layer):
    
    def __init__(self, timesteps=1000):
        super().__init__(name='f_encoder')
        self.timesteps = timesteps
        self.resnet = ResNet()
        self.freq_out = tfkl.Dense(128)
          
    def call(self, features):
        log_mel = features['log_mel'][:,:,:,tf.newaxis]
        resnet_out = self.resnet(log_mel)
        resnet_out = tf.reshape(resnet_out, [int(tf.shape(resnet_out)[0]), int(tf.shape(resnet_out)[1]), -1])
        freq_weights = self.freq_out(resnet_out)
        freq_weights = tf.nn.softplus(freq_weights) + 1e-3
        freq_weights = freq_weights / tf.reduce_sum(freq_weights, axis=-1, keepdims=True)
        f0s = self._compute_unit_midi(freq_weights)
        return resample(f0s, self.timesteps)
    
    def _compute_unit_midi(self, probs):
        # probs: [B, T, D]
        depth = int(tf.shape(probs)[-1])
        #unit_midi_bins = tf.constant(1.0 * tf.reshape(tf.range(depth), (1, 1, -1)) / depth, dtype=tf.float32)  # [1, 1, D]
        unit_midi_bins = tf.reshape(tf.range(depth), (1, 1, -1)) / depth
        unit_midi_bins = tf.cast(unit_midi_bins,tf.float32)
        f0_unit_midi = tf.reduce_sum(unit_midi_bins * probs, axis=-1, keepdims=True)  # [B, T, 1]
        return f0_unit_midi        