from tensorflow.keras import layers as tfkl

from dsp_utils.core import resample, midi_to_hz, hz_to_midi

from ddsp.spectral_ops import F0_RANGE
from ddsp.spectral_ops import LD_RANGE

from utilities import at_least_3d

class F0LoudnessPreprocessor(tfkl.Layer):
    """Resamples and scales 'f0_hz' and 'loudness_db' features. Used in the Supervised Setting."""

    def __init__(self, timesteps=250, **kwargs):
        super().__init__(**kwargs)
        self.timesteps = timesteps

    def call(self, inputs):

        # Downsample features, but do not update them in the dict
        f0_hz= self.resample(inputs["f0_hz"])
        loudness_db = self.resample(inputs["loudness_db"])

        # For NN training, scale frequency and loudness to the range [0, 1].
        # Log-scale f0 features. Loudness from [-1, 0] to [1, 0].
        f0_midi_scaled = hz_to_midi(f0_hz) / F0_RANGE
        ld_scaled = (loudness_db / LD_RANGE) + 1.0
        
        return {"f0_hz": at_least_3d(inputs["f0_hz"]),  # kept for the harmonic synth, convert to 3d here
                "f0_midi_scaled":f0_midi_scaled,        # used in the decoder in this form
                "ld_scaled":ld_scaled}                  # same as f0_midi_scaled

    def resample(self, x):
        x = at_least_3d(x)
        return resample(x, self.timesteps, method="linear")        

# TODO: fix the Encoder_f
# Downsample the f and l using the F0LoudnessPreprocessor
# Remove this class
class LoudnessPreprocessor(tfkl.Layer):

    def __init__(self, timesteps=250, **kwargs):
        super().__init__(**kwargs)
        self.timesteps = timesteps

    def call(self, inputs):

        loudness_db = inputs["loudness_db"]

        # Resample features to time_steps.
        loudness_db = self.resample(loudness_db)

        # For NN training, scale frequency and loudness to the range [0, 1].
        ld_scaled = (loudness_db / LD_RANGE) + 1.0

        return {"ld_scaled":ld_scaled}
        
    def resample(self, x):
        x = at_least_3d(x)
        return resample(x, self.timesteps, method="linear")


# TODO: delete??
class MidiF0LoudnessPreprocessor(tfkl.Layer):
    """Scales the loudness, converts scaled midi to hz and resamples. Used in the Unsupervised setting."""

    def __init__(self, timesteps=1000, **kwargs):
        super().__init__(**kwargs)
        self.timesteps = timesteps

    def call(self, inputs):
       
        loudness_db, f0_scaled = inputs["loudness_db"], inputs["f0_midi_scaled"]
               
        # Resample features to time_steps.
        f0_scaled = resample(f0_scaled, self.timesteps)
        loudness_db = resample(loudness_db, self.timesteps)
        
        # For NN training, scale frequency and loudness to the range [0, 1].
        ld_scaled = (loudness_db / LD_RANGE) + 1.0
        
        # ???????????????????????????
        # Convert scaled midi to hz for the synthesizer
        f0_hz = midi_to_hz(f0_scaled*F0_RANGE)
        
        f0_hz = resample(at_least_3d(f0_hz), 1000)
       
        return {"f0_hz":f0_hz, "loudness_db":loudness_db, "f0_midi_scaled":f0_scaled, "ld_scaled":ld_scaled}

    def resample(self, x):
        x = at_least_3d(x)
        return resample(x, self.timesteps, method="linear") 