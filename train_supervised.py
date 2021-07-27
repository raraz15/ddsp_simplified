import os, argparse

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from preprocessing import F0LoudnessPreprocessor
from encoders import SupervisedEncoder
from models import SupervisedAutoencoder
from decoders import DecoderWithoutLatent, DecoderWithLatent
from losses import SpectralLoss, MultiLoss
from callbacks import CustomWandbCallback, ModelCheckpoint
from dataloader import make_violin_set , make_supervised_dataset

SAMPLE_RATE = 16000

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Supervised Training Parameters.')
    #parser.add_argument('-d', '--dataset-path', type=str, help='Dataset Path')
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-t', '--preprocessing-time', type=int, help='Preprocessing timesteps', default=250)
    parser.add_argument('-d', '--decoder-time', type=int, help='Decoder timesteps', default=1000)
    parser.add_argument('-e', '--encoder', type=str, help='Use an encoder or not. (mfcc will be extracted)', default="False")
    parser.add_argument('-r', '--reverb', type=str, help='Add reverb', default="True")
    parser.add_argument('-l', '--loss', type=str, help='Loss Type', default='spectral')
    parser.add_argument('-n', '--name', type=str, help='Run Name')
    parser.add_argument('-w', '--wandb', type=str, help='Use Wandb', default="False")
    args = parser.parse_args()
    
    #train_set, validation_set, _ = make_supervised_dataset(args.dataset_path)
    train_set, validation_set, _ = make_violin_set(args.batch_size, args.encoder=="True")
    print('Dataset created.')
    preprocessor = F0LoudnessPreprocessor(timesteps=args.preprocessing_time)
    if args.encoder == "True":
        encoder = SupervisedEncoder()
        decoder = DecoderWithLatent(timesteps=args.decoder_time)
    else:
        encoder = None
        decoder = DecoderWithoutLatent(timesteps=args.decoder_time)
    loss = SpectralLoss() if args.loss == 'spectral' else MultiLoss() 
    tracker_names = ['spec_loss'] if loss.name=='spectral_loss' else ['spec_loss', 'perc_loss', 'total_loss']
    model = SupervisedAutoencoder(preprocessor=preprocessor,
                                encoder=encoder,
                                decoder=decoder,
                                loss_fn=loss,
                                tracker_names=tracker_names,
                                add_reverb=args.reverb)
    adam = Adam(learning_rate=ExponentialDecay(1e-3, decay_steps=10000, decay_rate=0.98))
    csv_logger = tf.keras.callbacks.CSVLogger("logs/{}.csv".format(args.name), separator=",", append=False)
    callbacks = [ModelCheckpoint(model, args.name), csv_logger]
    if args.wandb=="True":
        callbacks.append(CustomWandbCallback(args.name))
    model.compile(adam)
    print('Model Compiled.')
    history = model.fit(train_set, validation_data=validation_set, callbacks=callbacks, epochs=1000)