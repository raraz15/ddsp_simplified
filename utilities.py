from hashlib import md5
from typing import Dict, List

import numpy as np
from scipy.io.wavfile import write

import librosa

from tensorflow import newaxis

from utils.cache import Cache
from utils.midi_loader import MidiLoader


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


def generate_midi_features_examples(midi_data: Dict[str, np.ndarray], length_of_one_example: int) -> List[Dict[str, np.ndarray]]:

    keys = list(midi_data.keys())
    total_length = len(midi_data[keys[0]])

    examples = []
    for i in range(0, total_length-length_of_one_example, length_of_one_example):
        example = {}
        for key in keys:
            example[key] = midi_data[key][i:i+length_of_one_example]
        examples.append(example)

    return examples

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

def load_midi_track(path_to_midi_file: str, frame_rate: int, audio_length_seconds: float)->Dict[str, np.ndarray]:
    """
    Load a midi file.

    Loads a midi file and returns a dictionary mapping names of midi features
    to 1-d float32 numpy arrays with values of all possible midi features
    sampled at frame_rate

    Args:
        path_to_midi_file (str): absolute path to a MIDI file to be loaded
        frame_rate (int): rate at which midi features will be samples
        audio_length_seconds (float): full length (in seconds) of the corresponding audio file

    Returns:
        Dict[str, np.ndarray]: a dictionary mapping names of the midi features (as defined in MidiLoader class constants)
                               into corresponding numpy arrays
    """
    loader = MidiLoader()
    return loader.load(
        midi_file_name=path_to_midi_file,
        frame_rate=frame_rate,
        audio_length_seconds=audio_length_seconds
    )

def write_audio(audio, output_path, sample_rate=16000, normalize=False):
    assert '.wav' in output_path, 'Title must include .wav extension'
    if normalize:
        audio = librosa.util.normalize(audio)
    write(output_path, sample_rate, audio)