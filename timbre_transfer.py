import os
import yaml
from scipy.io.wavfile import write

import librosa

from preprocessing import F0LoudnessPreprocessor
from encoders import SupervisedEncoder
from decoders import DecoderWithoutLatent, DecoderWithLatent
from models import SupervisedAutoencoder, UnsupervisedAutoencoder
from losses import SpectralLoss, MultiLoss

from feature_extraction import process_track
from utilities import load_track

## -------------------------------------------- Timbre Transfer -------------------------------------------------

# scale loudness ?
def transfer_timbre_from_filepath(model, path, sample_rate=16000, pitch_shift=0, scale_loudness=0, **kwargs):
    track = load_track(path, sample_rate, pitch_shift=pitch_shift) 
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

def load_model_from_config(path):
    
    with open(path) as file:
        config = dict(yaml.load(file, Loader=yaml.FullLoader))
        
    data, train = config['data'], config['training']
    model_config, optim = config['model'], config['optimizer']   
    
    preprocessor = F0LoudnessPreprocessor(timesteps=data['preprocessing_time'])
    
    # Create the model and define the training 
    if model_config['encoder']:
        encoder = SupervisedEncoder()
        decoder = DecoderWithLatent(timesteps=model_config['decoder_time'])
    else:
        encoder = None
        decoder = DecoderWithoutLatent(timesteps=model_config['decoder_time'])

    loss = SpectralLoss() if config['loss']['type'] == 'spectral' else MultiLoss()
    
    if loss.name== 'spectral_loss':
        tracker_names = ['spec_loss']
        monitor = 'val_spec_loss'
    else:
        tracker_names = ['spec_loss', 'perc_loss', 'total_loss']
        monitor = 'val_total_loss'
    model = SupervisedAutoencoder(preprocessor=preprocessor,
                                encoder=encoder,
                                decoder=decoder,
                                loss_fn=loss,
                                tracker_names=tracker_names,
                                add_reverb=model_config['reverb'])
    model.load_weights(model_config['path'])
    return model