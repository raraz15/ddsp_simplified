import numpy as np

import librosa


def concat_dct(dcts):
    return {k:  np.array([dct[k] for dct in dcts]) for k in dcts[0].keys()}

def frame_generator(track, frame_size=64000):
    return track[:len(track)-len(track)%frame_size].reshape(-1, frame_size)

def load_track(path, sample_rate=16000, pitch_shift=0, normalize=True):
    track, _ = librosa.load(path, sr=sample_rate)
    if pitch_shift:
        track = librosa.effects.pitch_shift(track, sample_rate, n_steps=pitch_shift)
    if normalize:
        track = librosa.util.normalize(track)
    return track