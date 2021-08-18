import tensorflow as tf
from tensorflow.keras import layers as tfkl

from dsp_utils import core


class HarmonicSynthesizer(tfkl.Layer):
    def __init__(self,
               n_samples=64000,
               sample_rate=16000,
               scale_fn=core.exp_sigmoid,
               normalize_below_nyquist=True,
               amp_resample_method='window',
               name='harmonic'):

        super().__init__(name=name)
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.scale_fn = scale_fn
        self.normalize_below_nyquist = normalize_below_nyquist
        self.amp_resample_method = amp_resample_method

    def call(self, inputs):
        # get inputs
        amplitudes = inputs["amp_out"]
        harmonic_distribution = inputs["harmonic_out"]
        f0_hz = inputs["f0_hz"]
        
        # Scale the amplitudes for training
        if self.scale_fn is not None:
            amplitudes = self.scale_fn(amplitudes)
            harmonic_distribution = self.scale_fn(harmonic_distribution)
      
        # Bandlimit the harmonic distribution.
        if self.normalize_below_nyquist:
            n_harmonics = int(harmonic_distribution.shape[-1])
            harmonic_frequencies = core.get_harmonic_frequencies(f0_hz,
                                                               n_harmonics)
            
            harmonic_distribution = core.remove_above_nyquist(harmonic_frequencies,
                                                            harmonic_distribution,
                                                            self.sample_rate)

        # Normalize
        harmonic_distribution /= tf.reduce_sum(harmonic_distribution,
                                               axis=-1,
                                               keepdims=True)
        
        signal = core.harmonic_synthesis(
                            frequencies=f0_hz,
                            amplitudes=amplitudes,
                            harmonic_distribution=harmonic_distribution,
                            n_samples=self.n_samples,
                            sample_rate=self.sample_rate,
                            amp_resample_method=self.amp_resample_method)
        
        return signal
    
class FilteredNoiseSynthesizer(tfkl.Layer):

    def __init__(self,
               n_samples=64000,
               window_size=257,
               scale_fn=core.exp_sigmoid,
               initial_bias=-5.0,
               name='filtered_noise'):
        
        super().__init__(name=name)
        self.n_samples = n_samples
        self.window_size = window_size
        self.scale_fn = scale_fn
        self.initial_bias = initial_bias

    def call(self, inputs):
        
        magnitudes = inputs["noise_out"]
        if self.scale_fn is not None:
            magnitudes = self.scale_fn(magnitudes + self.initial_bias)
        
        batch_size = int(tf.shape(magnitudes)[0])
        signal = tf.random.uniform(
            [batch_size, self.n_samples], minval=-1.0, maxval=1.0)
        
        return core.frequency_filter(signal,
                                     magnitudes,
                                     window_size=self.window_size)

# Some things look a little changed from the original code    
class Reverb(tfkl.Layer):

    def __init__(self,
                 reverb_length=48000,
                 name='reverb'):
        
        super().__init__(name=name, trainable=True)
        self.reverb_length = reverb_length
        
    def build(self, input_shape):
        
        initializer = tf.random_normal_initializer(mean=0, stddev=1e-6)
        self.ir = tf.Variable(initial_value=initializer(shape=[self.reverb_length-1], dtype='float32'), trainable=True, name="ir")
        self.build = True

    def call(self, inputs):
        audio = inputs["audio_synth"]
        batch_size = int(tf.shape(audio)[0])
        
        ir = tf.repeat(self.ir[tf.newaxis,:], batch_size, axis=0)
        #ir = tf.tile(self.ir[tf.newaxis,:], [batch_size, 1])
        dry_mask = tf.zeros([int(tf.shape(ir)[0]), 1], tf.float32)
        ir = tf.concat([dry_mask, ir], axis=1)
        wet = core.fft_convolve(audio, ir, padding='same', delay_compensation=0)
        return audio + wet    