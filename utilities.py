import numpy as np
from scipy.io.wavfile import write

import librosa

from tensorflow import newaxis

def at_least_3d(x):
    """Optionally adds time, batch, then channel dimension."""
    x = x[newaxis] if not x.shape else x
    x = x[newaxis, :] if len(x.shape) == 1 else x
    x = x[:, :, newaxis] if len(x.shape) == 2 else x
    return x

def ensure_4d(x):
  """Add extra dimensions to make sure tensor has height and width."""
  if len(x.shape) == 2:
    return x[:, newaxis, newaxis, :]
  elif len(x.shape) == 3:
    return x[:, :, newaxis, :]
  else:
    return x    

def concat_dct(dcts):
    return {k:  np.array([dct[k] for dct in dcts]) for k in dcts[0].keys()}

def frame_generator(track, frame_size=64000):
    return track[:len(track)-len(track)%frame_size].reshape(-1, frame_size)

def load_track(path, sample_rate=16000, pitch_shift=0, normalize=False):
    """pitch_shift in semitones."""
    track, _ = librosa.load(path, sr=sample_rate)
    if pitch_shift:
        track = librosa.effects.pitch_shift(track, sr=sample_rate, n_steps=pitch_shift)
    if normalize:
        track = librosa.util.normalize(track)
    return track

def write_audio(audio, output_path, sample_rate=16000, normalize=False):
    assert '.wav' in output_path, 'Title must include .wav extension'
    if normalize:
        audio = librosa.util.normalize(audio)
    write(output_path, sample_rate, audio)