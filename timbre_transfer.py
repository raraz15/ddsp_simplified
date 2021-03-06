import os
import yaml
import argparse

from feature_extraction import process_track
from utilities import load_track, write_audio
from train_utils import make_supervised_model

## -------------------------------------------- Timbre Transfer -------------------------------------------------

def make_model_from_config(config):
    model = make_supervised_model(config)
    model.load_weights(config['model']['path'])
    return model 

# scale loudness ?
def transfer_timbre_from_path(model, path, sample_rate=16000,
                            pitch_shift=0, scale_loudness=0, 
                            normalize=False, **kwargs):
    track = load_track(path, sample_rate, pitch_shift=pitch_shift, normalize=normalize) 
    features = process_track(track, model=model, **kwargs)
    features["loudness_db"] +=  scale_loudness
    transfered_track = model.transfer_timbre(features)
    return transfered_track


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Timbre Transfer Parameters.')
    parser.add_argument('-c', '--config-path', type=str, required=True, help='Path to config file.')
    parser.add_argument('-a', '--audio-path', type=str, required=True, help='Path to audio file.')
    parser.add_argument('-o', '--output-dir', type=str, default='', help='Output audio directory.')
    parser.add_argument('-p', '--pitch-shift', type=int, default=0, help='Semi tones pitch shift.')
    parser.add_argument('-s', '--scale-loudness', type=int, default=0, help='Loudness scale.')
    args = parser.parse_args()    

    with open(args.config_path, 'r') as file:
        config = dict(yaml.load(file, Loader=yaml.FullLoader))    
    print(config['run_name'])
    model = make_model_from_config(config)
    print('Model loaded.')

    transfered_track = transfer_timbre_from_path(model,
                                                args.audio_path,
                                                sample_rate=config['data']['sample_rate'],
                                                pitch_shift=args.pitch_shift,
                                                scale_loudness=args.scale_loudness,
                                                mfcc=config['model']['encoder'],
                                                frame_rate=config['data']['preprocessing_time'],
                                                normalize=config['data']['normalize'])
    print('Timbre transfered.')
    print('Writing audio.')

    output_dir = args.output_dir
    if not output_dir:
        output_dir =  os.path.join('audio_clips','Outputs', 'DDSP_Violin', config['run_name'])
        #'DDSP_'+config['data']['instrument']
    os.makedirs(output_dir, exist_ok=True)
    input_name = os.path.basename(args.audio_path)
    output_name = os.path.splitext(input_name)[0]+'-timbre_transfered.wav'
    output_path = os.path.join(output_dir, output_name)
    
    write_audio(transfered_track,
                output_path,
                sample_rate=config['data']['sample_rate'],
                normalize=config['data']['normalize'])
                
    # Save the config next to the audio file
    with open(os.path.join(output_dir, 'model.yaml'), 'w') as f:
        yaml.dump(config, f)                                                                