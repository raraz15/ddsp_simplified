from hashlib import md5
from typing import Dict, List

import numpy as np
from scipy.io.wavfile import write

import librosa

from tensorflow import newaxis

from utils.cache import Cache


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


def _generate_cache_key_for_audio(path: str, sample_rate: int, pitch_shift: int, normalize: bool) -> str:
    return f"{md5(path.encode('utf-8')).hexdigest()}-{sample_rate}-{pitch_shift}-{normalize}"



def load_track(path, sample_rate=16000, pitch_shift=0, normalize=False):
    """
    Return uncompressed audio after sample rate conversion and some other manipulations.

    Additionally, uses caching.

    Args:
        path (str): path to the compressed file
        sample_rate (int): sample rate to convert the resulting audio to
        pitch_shift (int): pitch shift to apply, in semitones
        normalize (bool): whether to apply normalization

    Returns:
        np.ndarray: the raw uncompressed 1-channel audio with applicable processing.
    """

    cache_key = _generate_cache_key_for_audio(path, sample_rate, pitch_shift, normalize)
    if Cache.get_instance().has_numpy_array(cache_key):
        return Cache.get_instance().get_numpy_array(cache_key)

    track, _ = librosa.load(path, sr=sample_rate)
    if pitch_shift:
        track = librosa.effects.pitch_shift(track, sr=sample_rate, n_steps=pitch_shift)
    if normalize:
        track = librosa.util.normalize(track)

    Cache.get_instance().put_numpy_array(cache_key, track)

    return track

def load_midi_track(midi_file_name: str, frame_rate: int, feature_names: List[str]) -> Dict[str, np.ndarray]:
    pass

def write_audio(audio, output_path, sample_rate=16000, normalize=False):
    assert '.wav' in output_path, 'Title must include .wav extension'
    if normalize:
        audio = librosa.util.normalize(audio)
    write(output_path, sample_rate, audio)