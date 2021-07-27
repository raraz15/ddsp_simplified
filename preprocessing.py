import tensorflow as tf
from tensorflow.keras import layers as tfkl

from dsp_utils import core, spectral_ops

hz_to_midi = core.hz_to_midi
F0_RANGE = spectral_ops.F0_RANGE
LD_RANGE = spectral_ops.LD_RANGE


class F0LoudnessPreprocessor(tfkl.Layer):
    """Resamples and scales 'f0_hz' and 'loudness_db' features. Used in the Supervised Setting."""

    def __init__(self, timesteps=250, **kwargs):
        super().__init__(**kwargs)
        self.timesteps = timesteps

    def call(self, inputs):
       
        loudness_db, f0_hz = inputs["loudness_db"], inputs["f0_hz"]
               
        # Resample features to time_steps.
        f0_hz = core.resample(at_least_3d(f0_hz), 1000 , method="window") # to maintain consistency
        f0_hz_downsampled = core.resample(f0_hz, self.timesteps, method="window")
        loudness_db = core.resample(at_least_3d(loudness_db), self.timesteps, method="window")
                
        # For NN training, scale frequency and loudness to the range [0, 1].
        # Log-scale f0 features. Loudness from [-1, 0] to [1, 0].
        f0_midi_scaled = hz_to_midi(f0_hz_downsampled) / F0_RANGE
        ld_scaled = (loudness_db / LD_RANGE) + 1.0
        
        return {"f0_hz":f0_hz, "loudness_db":loudness_db, "f0_midi_scaled":f0_midi_scaled, "ld_scaled":ld_scaled}

    
class MidiF0LoudnessPreprocessor(tfkl.Layer):
    """Scales the loudness, converts scaled midi to hz and resamples. Used in the Unsupervised setting."""

    def __init__(self, timesteps=1000, **kwargs):
        super().__init__(**kwargs)
        self.timesteps = timesteps

    def call(self, inputs):
       
        loudness_db, f0_scaled = inputs["loudness_db"], inputs["f0_midi_scaled"]
               
        # Resample features to time_steps.
        f0_scaled = core.resample(at_least_3d(f0_scaled), self.timesteps)
        loudness_db = core.resample(at_least_3d(loudness_db), self.timesteps)
        
        # For NN training, scale frequency and loudness to the range [0, 1].
        ld_scaled = (loudness_db / LD_RANGE) + 1.0
        
        # Convert scaled midi to hz for the synthesizer
        f0_hz = core.midi_to_hz(f0_scaled*F0_RANGE)
        
        f0_hz = core.resample(at_least_3d(f0_hz), 1000)
       
        return {"f0_hz":f0_hz, "loudness_db":loudness_db, "f0_midi_scaled":f0_scaled, "ld_scaled":ld_scaled}

    
def at_least_3d(x):
    """Optionally adds time, batch, then channel dimension."""
    x = x[tf.newaxis] if not x.shape else x
    x = x[tf.newaxis, :] if len(x.shape) == 1 else x
    x = x[:, :, tf.newaxis] if len(x.shape) == 2 else x
    return x