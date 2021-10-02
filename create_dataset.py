import os
import glob
import argparse

import numpy as np

from sklearn.model_selection import train_test_split

from feature_extraction import extract_features_from_frames
from utilities import frame_generator, load_track


SAMPLE_RATE = 16000

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Supervised Dataset Parameters.')
    parser.add_argument('-a', '--audio-path', type=str, required=True, help='Path to the audio files.')
    parser.add_argument('-d', '--dataset-name', type=str, required=True, help='Name of the new dataset.')
    parser.add_argument('-f', '--dur-frame', type=int, default=4, help='Duration of a frame in seconds.')
    parser.add_argument('-n', '--normalize', action="store_true", help='Normalize audio files.')
    parser.add_argument('-c', '--conf-threshold', type=float, default=0.0, help='Confidence level threshold for the F0 extraction.')
    parser.add_argument('-m', '--mfcc', action="store_true", help='Extract mfcc features.')
    parser.add_argument('--mfcc-nfft', type=int, default=1024, help='Number of fft coefficients for mfcc extraction.')    
    args = parser.parse_args()

    frames = []
    for file in glob.glob(args.audio_path+'/*.mp3') + glob.glob(args.audio_path+ '/*.wav'):
        print(os.path.basename(file))  
        track = load_track(file, sample_rate=SAMPLE_RATE, normalize=args.normalize)
        frames.append(frame_generator(track, args.dur_frame*SAMPLE_RATE))
    frames = np.concatenate(frames, axis=0)   
    
    trainX, valX = train_test_split(frames)
    print('Train set size: {}\nVal set size: {}'.format(len(trainX),len(valX)))

    train_features = extract_features_from_frames(trainX, mfcc=args.mfcc, sample_rate=SAMPLE_RATE,
                                                conf_threshold=args.conf_threshold, mfcc_nfft=args.mfcc_nfft)
    val_features = extract_features_from_frames(valX, mfcc=args.mfcc, sample_rate=SAMPLE_RATE,
                                                conf_threshold=args.conf_threshold, mfcc_nfft=args.mfcc_nfft)

    ds_dir = os.path.join('datasets', args.dataset_name)
    os.makedirs(ds_dir, exist_ok=True)
    np.save(os.path.join(ds_dir, 'train.npy'), train_features)
    np.save(os.path.join(ds_dir, 'val.npy'), val_features)