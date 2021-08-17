import glob
import numpy as np

from tensorflow.data import Dataset
from sklearn.model_selection import train_test_split

from feature_extraction import extract_features_from_frames
from utilities import frame_generator, load_track


# Normalize each track or normalize each frame ?????
def make_supervised_dataset(path, mfcc=False, batch_size=32, sample_rate=16000, normalize=False):
    """Loads all the mp3 files in the path, creates frames and extracts features."""
    frames = []
    for file in glob.glob(path+'/*.mp3'):    
        track = load_track(file, sample_rate=sample_rate, normalize=normalize)
        # create 4 seconds long frames
        frames.append(frame_generator(track, 4*sample_rate))     
    frames = np.concatenate(frames, axis=0)   
    trainX, valX = train_test_split(frames)
    print('Train set size: {}'.format(len(trainX)))
    print('Val set size: {}'.format(len(valX)))
    train_features = extract_features_from_frames(trainX, mfcc=mfcc, sample_rate=sample_rate)
    val_features = extract_features_from_frames(valX, mfcc=mfcc, sample_rate=sample_rate)
    return _make_dataset(train_features, batch_size), _make_dataset(val_features, batch_size), None

#def make_unsupervised_dataset(path, batch_size=32, sr=16000, normalize=False):

def _make_dataset(features, batch_size=32, seed=None):
    features = Dataset.from_tensor_slices(features)
    features = features.shuffle(len(features)*2, seed, True) # shuflle at each iteration
    features = features.batch(batch_size)
    features = features.prefetch(1) # preftech 1 batch
    return features

# ------------------------------- maybe for Nsynth --------------------------

#import tensorflow as tf
#import tensorflow_datasets as tfds
#from tensorflow.keras.utils import Sequence

#def make_violin_set(batch_size=32, mfcc=False):
#    frames = np.load(open("audio_clips/Violin/npy/violin_frames.npy","rb"))
#    trainX, valX = train_test_split(frames)
#    train_features = extract_features(trainX, mfcc)
#    val_features = extract_features(valX, mfcc)
#    return _make_dataset(train_features, batch_size), _make_dataset(val_features, batch_size), None  

#def make_violin_set():
#    frames = np.load(open("audio_clips/violin/npy/violin_frames.npy","rb"))
#    trainX, valX = train_test_split(frames)
#    trainX =  tf.data.Dataset.from_tensor_slices(concat_dct(list(map(loudness_and_f0_extractor, trainX))))
#    valX =  tf.data.Dataset.from_tensor_slices(concat_dct(list(map(loudness_and_f0_extractor, valX))))
#    return trainX.prefetch(tf.data.experimental.AUTOTUNE), valX.prefetch(tf.data.experimental.AUTOTUNE), None

#def preprocess_ex(ex, mfcc=False, log_mel=False):
#    dct = {'pitch': ex['pitch'],
#          'audio': ex['audio'],   
#          'instrument_source': ex['instrument']['source'],
#          'instrument_family': ex['instrument']['family'],
#          'instrument': ex['instrument']['label'],
#          'f0_hz': ex['f0']['hz'],
#          'f0_confidence': ex['f0']['confidence'],
#          'loudness_db': ex['loudness']['db']}
#    if log_mel:
#        dct['log_mel'] = compute_logmel(ex['audio'])
#    if mfcc:
#        dct["mfcc"] = compute_mfcc(ex['audio'], lo_hz=20.0, hi_hz=8000.0,
#                    fft_size=1024, mel_bins=128, mfcc_bins=30)
#    return dct

#def make_datasets_original(batch_size, mfcc, log_mel, percent=100):
#    train_set = tfds.load('nsynth/gansynth_subset.f0_and_loudness:2.3.3',download=False,
#                      shuffle_files=True,batch_size=batch_size,split="train[:{}%]".format(percent)).map(lambda x: preprocess_ex(x, mfcc, log_mel))
#    val_set = tfds.load('nsynth/gansynth_subset.f0_and_loudness:2.3.3',download=False,
#                      shuffle_files=True,batch_size=batch_size,split="valid[:{}%]".format(percent)).map(lambda x: preprocess_ex(x, mfcc, log_mel))
#    test_set = tfds.load('nsynth/gansynth_subset.f0_and_loudness:2.3.3',download=False,
#                      shuffle_files=True,batch_size=batch_size,split="test[:{}%]".format(percent)).map(lambda x: preprocess_ex(x, mfcc, log_mel))
#    return train_set,val_set,test_set
    

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