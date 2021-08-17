import yaml
import argparse

from feature_extraction import process_track
from utilities import load_track, write_audio
from train_supervised import make_supervised_model

## -------------------------------------------- Timbre Transfer -------------------------------------------------

def load_model_from_config(config):
    model = make_supervised_model(config)
    model.load_weights(config['model']['path'])
    return model   

# scale loudness ?
def transfer_timbre_from_path(model, path, sample_rate=16000, pitch_shift=0,
                            scale_loudness=0, normalize=False, **kwargs):
    track = load_track(path, sample_rate, pitch_shift=pitch_shift, normalize=normalize) 
    features = process_track(track, model=model, **kwargs)
    features["loudness_db"] +=  scale_loudness
    transfered_track = model.transfer_timbre(features)
    return transfered_track

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Timbre Transfer Parameters.')
    parser.add_argument('-c', '--config-path', type=str, required=True, help='Path to config file.')
    parser.add_argument('-a', '--audio-path', type=str, required=True, help='Path to audio file.')
    parser.add_argument('-o', '--output-path', type=str, required=True, help='Output audio path.')
    parser.add_argument('-p', '--pitch-shift', type=int, default=0, help='Semi tones pitch shift.')
    parser.add_argument('-s', '--scale-loudness', type=int, default=0, help='Loudness scale.')
    parser.add_argument('-n', '--normalize', default=False, action='store_true', help='Normalize audio.')
    args = parser.parse_args()    

    with open(args.config_path) as file:
        config = dict(yaml.load(file, Loader=yaml.FullLoader))    
    model = load_model_from_config(config)
    print('Model loaded.')
    transfered_track = transfer_timbre_from_path(model,
                                                args.audio_path,
                                                sample_rate=config['data']['sample_rate'],
                                                pitch_shift=args.pitch_shift,
                                                scale_loudness=args.scale_loudness,
                                                mfcc=config['model']['encoder'],
                                                frame_rate=config['data']['preprocessing_time'],
                                                #log_mel=config[]
                                                normalize=args.normalize)
    print('Timbre transferred.')
    print('Writing audio.')                                                
    write_audio(transfered_track,
                args.output_path,
                sample_rate=config['data']['sample_rate'],
                normalize=args.normalize)                                                