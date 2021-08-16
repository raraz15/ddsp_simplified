import os
from scipy.io.wavfile import write

import librosa

from feature_extraction import process_track
from utilities import load_track

## -------------------------------------------- Timbre Transfer -------------------------------------------------

# scale loudness ?
def transfer_timbre_from_filepath(model, path, sample_rate=16000, pitch_shift=0, scale_loudness=0, **kwargs):
    track, _ = load_track(path, sample_rate, pitch_shift=pitch_shift) 
    features = process_track(track, model=model, **kwargs)
    features["loudness_db"] +=  scale_loudness
    transfered_track = model.transfer_timbre(features)
    return transfered_track

def write_audio(audio, title, RUN_NAME, sample_rate=16000, normalize=True):
    assert '.wav' in title, 'Title must include .wav extension'
    output_path = os.path.join('audio_clips','outputs', RUN_NAME, title)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if normalize:
        audio = librosa.util.normalize(audio)
    write(output_path, sample_rate, audio)