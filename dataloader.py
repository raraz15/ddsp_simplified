import glob
import os.path
from typing import Dict, Optional, List

import numpy as np

from tensorflow.data import Dataset
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

from feature_extraction import extract_features_from_frames, feature_extractor
from utilities import frame_generator, load_track, get_raw_midi_features_from_file, generate_midi_features_examples, concat_dct

MIDI_FILE_EXTENSION = 'MID'

def _make_dataset(features, batch_size=32, seed=None):
    features = Dataset.from_tensor_slices(features)
    features = features.shuffle(len(features)*2, seed, True) # shuflle at each iteration
    features = features.batch(batch_size)
    features = features.prefetch(1) # preftech 1 batch
    return features

# -------------------------------------------- Supervised Dataset -------------------------------------------------

def guess_midi_file_name_by_audio_file_name(audio_file_name: str) -> str:
    return os.path.splitext(audio_file_name)[0] + '.' + MIDI_FILE_EXTENSION


def _filter_midi_features_data(raw_midi_features_data: Dict[str, np.array], feature_names_to_leave: List[str]) -> Dict[str, np.array]:
    """
    Get a dict of feature names and return its version with only requested features left.
    """
    return {feature_name: raw_midi_features_data[feature_name] for feature_name in feature_names_to_leave}



def make_supervised_dataset(path, mfcc=False, batch_size=32, sample_rate=16000,
                            normalize=False, conf_threshold=0.0, mfcc_nfft=1024,
                            frame_rate: int = 250,
                            midi_feature_names: Optional[List[str]] = None
                            ):
    """Loads all the mp3 and (optionally) midi files in the path, creates frames and extracts features."""

    ordered_audio_frames = []
    ordered_midi_features_frames = []

    length_of_example_seconds = 4.0

    # this one is essentially just syntactic sugar
    need_midi = True if midi_feature_names is not None else False

    KEY_AUDIO = 'audio'
    KEY_MIDI = 'midi'

    for audio_file_name in glob.glob(path+'/*.mp3'):

        audio_data = load_track(audio_file_name, sample_rate=sample_rate, normalize=normalize)
        generated_audio_frames = frame_generator(audio_data, int(length_of_example_seconds * sample_rate))
        ordered_audio_frames.extend(generated_audio_frames)  # create 4 seconds long frames\

        if need_midi:
            midi_file_name = guess_midi_file_name_by_audio_file_name(audio_file_name)
            raw_midi_features_data = get_raw_midi_features_from_file(midi_file_name, frame_rate, audio_data.shape[0] / sample_rate)
            raw_midi_features_data = _filter_midi_features_data(raw_midi_features_data, midi_feature_names)

            generated_midi_feature_examples = generate_midi_features_examples(raw_midi_features_data, int(length_of_example_seconds * frame_rate))
            ordered_midi_features_frames.extend(generated_midi_feature_examples)

    combined_frames = []
    if need_midi:
        assert len(ordered_audio_frames) == len(ordered_midi_features_frames)
        for (audio_frame, midi_features_frame) in zip(ordered_audio_frames, ordered_midi_features_frames):
            combined_frames.append({
                KEY_AUDIO: audio_frame,
                KEY_MIDI: midi_features_frame
            })
    else:
        for audio_frame in ordered_audio_frames:
            combined_frames.append({
                KEY_AUDIO: audio_frame,
            })

    trainX, valX = train_test_split(combined_frames)

    train_shuffled_audio_frames = [x[KEY_AUDIO] for x in trainX]
    val_shuffled_audio_frames = [x[KEY_AUDIO] for x in valX]

    # audio_and_midi_features_frames = np.concatenate(audio_and_midi_features_frames, axis=0)

    print('Train set size: {}\nVal set size: {}'.format(len(trainX),len(valX)))
    train_audio_and_audio_features = extract_features_from_frames(train_shuffled_audio_frames, mfcc=mfcc, sample_rate=sample_rate,
                                                                  conf_threshold=conf_threshold, mfcc_nfft=mfcc_nfft)
    val_audio_and_audio_features = extract_features_from_frames(val_shuffled_audio_frames, mfcc=mfcc, sample_rate=sample_rate,
                                                                conf_threshold=conf_threshold, mfcc_nfft=mfcc_nfft)

    if need_midi:
        train_shuffled_midi_frames = [x[KEY_MIDI] for x in trainX]
        val_shuffled_midi_frames = [x[KEY_MIDI] for x in valX]

        train_shuffled_midi_frames = concat_dct(train_shuffled_midi_frames)
        val_shuffled_midi_frames = concat_dct(val_shuffled_midi_frames)

        combined_train_features = {
            **train_audio_and_audio_features,
            **train_shuffled_midi_frames
        }

        combined_val_features = {
            **val_audio_and_audio_features,
            **val_shuffled_midi_frames
        }
    else:
        combined_train_features = train_audio_and_audio_features
        combined_val_features = val_audio_and_audio_features

    train_dataset = _make_dataset(combined_train_features, batch_size)
    val_dataset = _make_dataset(combined_val_features, batch_size)

    return train_dataset, val_dataset, None

# -------------------------------------------- Unsupervised Datasets ----------------------------------------------

def make_unsupervised_dataset(path, batch_size=32, sample_rate=16000, normalize=False, frame_rate=250):
    frames = []
    for file in glob.glob(path+'/*.mp3'):    
        track = load_audio_track(file, sample_rate=sample_rate, normalize=normalize)
        frames.append(frame_generator(track, 4*sample_rate)) # create 4 seconds long frames     
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


#import tensorflow as tf

#from tensorflow.keras.utils import Sequence

#def make_datasets(batch_size, percent=100):
#    datasets = [create_tf_dataset_from_npzs(glob.glob("data/{}/*.npz".format(folder)), batch_size, percent) for 
#                         folder in ["train","validation","test"]]
#    return datasets

#def create_tf_dataset_from_npzs(npzs, batch_size, percent):
#    def generator(dataset_generator):
#        while True:
#            for batch in dataset_generator:
#                yield batch
#    npzs = npzs[:int(len(npzs)*percent/100)]
#    data_generator = DataGenerator(npzs, batch_size)
#    generator_func = lambda: generator(data_generator)
#    sample = data_generator[0]
#    output_shapes = {k:sample[k].shape for k in sample}
#    output_types = {k:sample[k].dtype for k in sample}
#    dataset = Dataset.from_generator(generator_func, output_shapes=output_shapes, output_types=output_types)
#    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
#    dataset.my_len = len(data_generator)
#    return dataset

#class DataGenerator(Sequence):
#    def __init__(self, list_IDs, batch_size=64, shuffle=True, **kwargs):
#        super().__init__(**kwargs)
#        self.batch_size = batch_size
#        self.list_IDs = list_IDs[:-(len(list_IDs)%batch_size)]
#        self.shuffle = shuffle
#        self.reset()
#
#    def __len__(self):
#        'Denotes the number of batches per epoch'
#        return int(np.floor(len(self.list_IDs) / self.batch_size))
#
#    def __getitem__(self, index):
#        'Generate one batch of data'
#        # Generate indexes of the batch
#        if (index+1)*self.batch_size >= len(self.indexes):
#            self.reset()
#        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#
#        # Find list of IDs
#        list_IDs_temp = [self.list_IDs[k] for k in indexes]
#
#        # Generate data
#        X = self.__data_generation(list_IDs_temp)
#
#        return X
#
#    def reset(self):
#        'Updates indexes after each epoch'
#        self.indexes = np.arange(len(self.list_IDs))
#        if self.shuffle == True:
#            np.random.shuffle(self.indexes)
#
#    def __data_generation(self, list_IDs_temp):
#        batch = {}
#        arrays = [np.load(ID) for ID in list_IDs_temp]
#        for file in arrays[0].files:
#            batch[file] = np.array([arr[file] for arr in arrays])
#        return batch