import os
import glob
import numpy as np

from tensorflow.data import Dataset
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

from feature_extraction import extract_features_from_frames, feature_extractor
from utilities import frame_generator, load_track

# This script is for creating supervised and unsupervised datasets during training
# without exporting the datasets or laoding existing ones.
# If you require a dataset to be created and exported, use create_dataset.py

FRAME_LEN = 4

def load_dataset(dataset_dir, batch_size=32):
    train_path = os.path.join(dataset_dir, 'train.npy')
    val_path = os.path.join(dataset_dir, 'val.npy')
    train_features = np.load(train_path, allow_pickle=True).item()
    val_features = np.load(val_path, allow_pickle=True).item()
    return _make_dataset(train_features, batch_size), _make_dataset(val_features, batch_size), None


def _make_dataset(features, batch_size=32, seed=None):
    """Creates a dataset from extracted features."""
    features = Dataset.from_tensor_slices(features)
    features = features.shuffle(len(features)*2, seed, True) # shuflle at each iteration
    features = features.batch(batch_size)
    features = features.prefetch(1) # preftech 1 batch
    return features

# -------------------------------------------- Supervised Dataset -------------------------------------------------

def make_supervised_dataset(path, mfcc=False, batch_size=32, sample_rate=16000,
                            normalize=False, conf_threshold=0.0, mfcc_nfft=1024):
    """Loads all the mp3 files in the path, creates frames and extracts features."""

    frames = []
    for file in glob.glob(path+'/*.mp3'):    
        track = load_track(file, sample_rate=sample_rate, normalize=normalize)
        frames.append(frame_generator(track, FRAME_LEN*sample_rate))   
    frames = np.concatenate(frames, axis=0)   
    trainX, valX = train_test_split(frames)
    print('Train set size: {}\nVal set size: {}'.format(len(trainX),len(valX)))
    train_features = extract_features_from_frames(trainX, mfcc=mfcc, sample_rate=sample_rate,
                                                conf_threshold=conf_threshold, mfcc_nfft=mfcc_nfft)
    val_features = extract_features_from_frames(valX, mfcc=mfcc, sample_rate=sample_rate,
                                                conf_threshold=conf_threshold, mfcc_nfft=mfcc_nfft)
    return _make_dataset(train_features, batch_size), _make_dataset(val_features, batch_size), None

# -------------------------------------------- Unsupervised Datasets ----------------------------------------------
# TODO: make functional

def make_unsupervised_dataset(path, batch_size=32, sample_rate=16000, normalize=False, frame_rate=250):
    frames = []
    for file in glob.glob(path+'/*.mp3'):    
        track = load_track(file, sample_rate=sample_rate, normalize=normalize)
        frames.append(frame_generator(track, FRAME_LEN*sample_rate))     
    frames = np.concatenate(frames, axis=0)   
    trainX, valX = train_test_split(frames)
    train_features = extract_features_from_frames(trainX, f0=False, mfcc=True, log_mel=True,
                                                sample_rate=sample_rate, frame_rate=frame_rate)
    val_features = extract_features_from_frames(valX, f0=False, mfcc=True, log_mel=True,
                                                sample_rate=sample_rate, frame_rate=frame_rate)
    return _make_dataset(train_features, batch_size), _make_dataset(val_features, batch_size), None    

#/scratch/users/hbalim15/tensorflow_datasets/nsynth/
def make_nsynth_dataset(batch_size, path='nsynth/gansynth_subset.f0_and_loudness:2.3.3'):
    split=["train[:80%]", 'validation[:10%]', 'test[:10%]']
    train_set, val_set, test_set = tfds.load(path,
                                            download=False,
                                            shuffle_files=True,
                                            batch_size=batch_size,
                                            split=split).prefetch(1).map(lambda x: preprocess_ex(x))
    return train_set, val_set, test_set

def preprocess_ex(ex):   
    dct = {'pitch': ex['pitch'],  
          'f0_hz': ex['f0']['hz'],
          'loudness_db': ex['loudness']['db']}
    dct.update(feature_extractor(ex['audio'], sample_rate=16000, frame_rate=250,
                        f0=False, mfcc=True, log_mel=True))
    return dct  