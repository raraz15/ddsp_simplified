import tensorflow as tf
from tensorflow.keras import layers as tfkl

from dsp_utils.core import resample

from utilities import at_least_3d

class MLP(tf.keras.Sequential):
    """Stack Dense -> LayerNorm -> Leaky ReLU layers."""  
    def __init__(self, output_dim=256, n_layers=2, **kwargs):
        layers = []
        for _ in range(n_layers):
            layers.append(tfkl.Dense(output_dim, activation=None))
            # even though there is a layernorm in the paper,
            # with it the model in unstable
            #layers.append(tfkl.LayerNormalization())
            layers.append(tfkl.LeakyReLU())
        super().__init__(layers, **kwargs) 

class DecoderWithoutLatent(tfkl.Layer):
    """ Decoder class for loudness and F0. Used in the Supervised setting."""
    
    def __init__(self, rnn_channels=512, mlp_channels=512,
                 harmonic_out_channel=100, noise_out_channel=65,
                 layers_per_stack=3, timesteps=1000,
                 **kwargs):
        
        super().__init__(**kwargs)

        self.MLP_f0 = MLP(mlp_channels, layers_per_stack)
        self.MLP_l = MLP(mlp_channels, layers_per_stack)

        self.rnn = tfkl.GRU(rnn_channels, return_sequences=True)
        self.MLP_rnn = MLP(mlp_channels, layers_per_stack)

        self.dense_harmonic = tfkl.Dense(harmonic_out_channel)
        self.dense_amp = tfkl.Dense(1)
        self.dense_noise = tfkl.Dense(noise_out_channel)

        self.timesteps = timesteps

    def call(self, inputs):
        
        x_f0 = self.MLP_f0(inputs['f0_scaled'])
        x_l = self.MLP_l(inputs['ld_scaled'])
        inputs = [x_f0, x_l]
        
        x = tf.concat(inputs, axis=-1)
        x = self.rnn(x)
        x = tf.concat(inputs + [x], axis=-1) 
        x = self.MLP_rnn(x)
                       
        # Synthesizer Parameters
        amp_out = self.dense_amp(x)
        harmonic_out = self.dense_harmonic(x)
        noise_out = self.dense_noise(x)
        
        # Upsampling to the audio rate here.
        return {'amp_out': self.resample(amp_out),
                'harmonic_out': self.resample(harmonic_out),
                'noise_out': self.resample(noise_out)}
    
    def resample(self, x):
        x = at_least_3d(x)
        return resample(x, self.timesteps, method='window')
    
class DecoderWithLatent(tfkl.Layer):
    """Decoder class for Z, F0 and l. Used in the Unsupervised Setting."""
    
    def __init__(self, rnn_channels=512, mlp_channels=512, layers_per_stack=3,
                 harmonic_out_channel=100, noise_out_channel=65,
                 timesteps=1000, **kwargs):
        
        super().__init__(**kwargs)
        
        self.MLP_f0 = MLP(mlp_channels, layers_per_stack)
        self.MLP_l = MLP(mlp_channels, layers_per_stack)
        self.MLP_z = MLP(mlp_channels, layers_per_stack)

        self.rnn = tfkl.GRU(rnn_channels, return_sequences=True)
        self.MLP_rnn = MLP(mlp_channels, layers_per_stack)
        
        self.dense_harmonic = tfkl.Dense(harmonic_out_channel)
        self.dense_amp = tfkl.Dense(1)
        self.dense_noise = tfkl.Dense(noise_out_channel)
        
        self.timesteps = timesteps

    def call(self, inputs):
        
        x_f0 = self.MLP_f0(inputs['f0_scaled'])
        x_l = self.MLP_l(inputs['ld_scaled'])
        x_z = self.MLP_z(inputs['z'])
        
        inputs = [x_f0, x_l, x_z]
        x = tf.concat(inputs, axis=-1)
          
        x = self.rnn(x)
        
        x = tf.concat(inputs + [x], axis=-1)
        x = self.MLP_rnn(x)
                       
        # Parameters of the synthesizers
        amp_out = self.dense_amp(x)
        harmonic_out = self.dense_harmonic(x)
        noise_out = self.dense_noise(x)
    
        return {'amp_out': self.resample(amp_out),
                'harmonic_out': self.resample(harmonic_out),
                'noise_out': self.resample(noise_out)}
    
    def resample(self, x):
        x = at_least_3d(x)
        return resample(x, self.timesteps, method='window')