import glob
import numpy as np

from tensorflow.data import Dataset
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

from feature_extraction import extract_features_from_frames, feature_extractor
from utilities import frame_generator, load_track


def _make_dataset(features, batch_size=32, seed=None):
    features = Dataset.from_tensor_slices(features)
    features = features.shuffle(len(features)*2, seed, True) # shuflle at each iteration
    features = features.batch(batch_size)
    features = features.prefetch(1) # preftech 1 batch
    return features

# -------------------------------------------- Supervised Dataset -------------------------------------------------

def make_supervised_dataset(path, mfcc=False, batch_size=32, sample_rate=16000, normalize=False, conf_threshold=0.0):
    """Loads all the mp3 files in the path, creates frames and extracts features."""
    frames = []
    for file in glob.glob(path+'/*.mp3'):    
        track = load_track(file, sample_rate=sample_rate, normalize=normalize)
        frames.append(frame_generator(track, 4*sample_rate)) # create 4 seconds long frames     
    frames = np.concatenate(frames, axis=0)   
    trainX, valX = train_test_split(frames)
    print('Train set size: {}\nVal set size: {}'.format(len(trainX),len(valX)))
    train_features = extract_features_from_frames(trainX, mfcc=mfcc,sample_rate=sample_rate, conf_threshold=conf_threshold)
    val_features = extract_features_from_frames(valX, mfcc=mfcc, sample_rate=sample_rate, conf_threshold=conf_threshold)
    return _make_dataset(train_features, batch_size), _make_dataset(val_features, batch_size), None

# -------------------------------------------- Unsupervised Datasets ----------------------------------------------

def make_unsupervised_dataset(path, batch_size=32, sample_rate=16000, normalize=False, conf_threshold=0.0):
    frames = []
    for file in glob.glob(path+'/*.mp3'):    
        track = load_track(file, sample_rate=sample_rate, normalize=normalize)
        frames.append(frame_generator(track, 4*sample_rate)) # create 4 seconds long frames     
    frames = np.concatenate(frames, axis=0)   
    trainX, valX = train_test_split(frames)
    train_features = extract_features_from_frames(trainX, f0=False, mfcc=True, log_mel=True,
                                                sample_rate=sample_rate, conf_threshold=conf_threshold)
    val_features = extract_features_from_frames(valX, f0=False, mfcc=True, log_mel=True,
                                                sample_rate=sample_rate, conf_threshold=conf_threshold)
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