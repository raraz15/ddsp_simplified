import numpy as np

import tensorflow as tf
from tensorflow.keras import layers as tfkl
import tensorflow_addons as tfa

from dsp_utils.core import resample, hz_to_midi
from dsp_utils.spectral_ops import F0_RANGE

from resnet import ResNet
from utilities import at_least_3d

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
    
    def __init__(self, rnn_channels=512, z_dims=32, timesteps=250):
        
        super().__init__(name='UnsupervisedEncoder')

        self.timesteps = timesteps
        
        self.encoder_z = Encoder_z(rnn_channels, z_dims)
        self.encoder_f = Encoder_f() #timesteps=timesteps
        #l is extracted in the dataset, normalized in the preprocessor
        
    # f0_midi_scaled ???????????
    def call(self, features):
        
        z = self.encoder_z(features)

        print('z: {}'.format(z.shape))

        ld_scaled = features['ld_scaled']

        print('ld_scaled: {}'.format(ld_scaled.shape))
        f0_hz = self.encoder_f(features)

        print('f0_hz: {}'.format(f0_hz))

        f0_midi_scaled = hz_to_midi(f0_hz) / F0_RANGE         
        
        return {'z': z,
                'f0_hz': f0_hz,
                'f0_midi_scaled': f0_midi_scaled,
                'ld_scaled': ld_scaled}
               

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

# # upsampling fails
# MIDI Scaling??
# compute unit midi??
class Encoder_f(tfkl.Layer):
    
    def __init__(self, timesteps=250):
        super().__init__(name='f_encoder')
        self.timesteps = timesteps
        self.resnet = ResNet()

        self.reshape0 = tfkl.Reshape((125, 1, 8*1024))

        self.freq_out = tfkl.Dense(128)

        self.freq_scale = tf.convert_to_tensor(440* 2**((np.arange(0,128)-69)/12), dtype=tf.float32)
          
    def call(self, features):

        log_mel = features['log_mel'][:,:,:,tf.newaxis] # N, T, 229, 1
        resnet_out = self.resnet(log_mel)

        # # Collapse the frequency dimension.
        # tf reshape is a terrible thing and I've avoided it with a reshape layer
        resnet_out = self.reshape0(resnet_out)

        # should be T, 1, 128
        freq_weights = self.freq_out(resnet_out)
        print('freq_weights: {}'.format(freq_weights.shape))

        # Not sure about the order because,
        # 1) upsampling should be done once here and once in decoder?
        # 2) then where is the softplus and normalize ?
        # 3) what is _compute_unit_midi ?

        freq_weights = self.resample(freq_weights)
        print('freq_weights resampled: {}'.format(freq_weights.shape))        


        # Softplus and normalize
        freq_weights = tf.nn.softplus(freq_weights) + 1e-3
        freq_weights /= tf.reduce_sum(freq_weights, axis=-1, keepdims=True)

        print('freq_weights soft, norm: {}'.format(freq_weights.shape))

        f0s = tf.math.reduce_sum(freq_weights * self.freq_scale,axis=-1)
        print('f0s: {}'.format(f0s.shape))


        #f0s = self.resample(f0s)
        #print('f0s resampled: {}'.format(f0s.shape))        

        return f0s
    
    #def _compute_unit_midi(self, probs):
    #    # probs: [B, T, D]
    #    depth = int(tf.shape(probs)[-1])
    #    #unit_midi_bins = tf.constant(1.0 * tf.reshape(tf.range(depth), (1, 1, -1)) / depth, dtype=tf.float32)  # [1, 1, D]
    #    unit_midi_bins = tf.reshape(tf.range(depth), (1, 1, -1)) / depth
    #    unit_midi_bins = tf.cast(unit_midi_bins,tf.float32)
    #    f0_unit_midi = tf.reduce_sum(unit_midi_bins * probs, axis=-1, keepdims=True)  # [B, T, 1]
    #    return f0_unit_midi    
            
    def resample(self, x):
        x = at_least_3d(x)
        return resample(x, self.timesteps, method="window")        