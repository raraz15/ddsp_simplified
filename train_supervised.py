import os, argparse, yaml

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from preprocessing import F0LoudnessPreprocessor
from encoders import SupervisedEncoder
from models import SupervisedAutoencoder
from decoders import DecoderWithoutLatent, DecoderWithLatent
from losses import SpectralLoss
from callbacks import ModelCheckpoint, CustomWandbCallback 
from dataloader import make_supervised_dataset


# TODO: use epoch name in model saving

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Supervised Training Parameters.')
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to config file.')   
    args = parser.parse_args()

    # Read the config
    with open(args.path) as file:
        config = dict(yaml.load(file, Loader=yaml.FullLoader))
    data, train = config['data'], config['training']
    model_config, optim = config['model'], config['optimizer']
    
    # Create the datasets and the preprocessor
    train_set, validation_set, _ = make_supervised_dataset(data['path'],
                                                    mfcc=model_config['encoder'],
                                                    batch_size=train['batch_size'],
                                                    sample_rate=data['sample_rate'])
    print('Dataset created.')
    preprocessor = F0LoudnessPreprocessor(timesteps=data['preprocessing_time'])
    
    # Create the model and define the training 
    if model_config['encoder']:
        encoder = SupervisedEncoder()
        decoder = DecoderWithLatent(timesteps=model_config['decoder_time'])
    else:
        encoder = None
        decoder = DecoderWithoutLatent(timesteps=model_config['decoder_time'])

    loss = SpectralLoss(logmag_weight=config['loss']['logmag_weight'])
    tracker_names = ['spec_loss']
    monitor = 'val_spec_loss'
        
    model = SupervisedAutoencoder(preprocessor=preprocessor,
                                encoder=encoder,
                                decoder=decoder,
                                loss_fn=loss,
                                tracker_names=tracker_names,
                                add_reverb=model_config['reverb'])
    optimizer = Adam(learning_rate=ExponentialDecay(optim['lr'],
                                decay_steps=optim['decay_steps'],
                                decay_rate=optim['decay_rate']))


    # Model Saving and Experiment Tracking
    # It looks ugly, but is necessary
    if model_config['dir']: # if dir specified, save there
        model_dir = model_config['dir']
        if not config['wandb']:
            callbacks = [ModelCheckpoint(save_dir=model_dir, monitor=monitor)]
        else:
            callbacks = [ModelCheckpoint(save_dir=model_dir, monitor=monitor),
                        CustomWandbCallback(config)]
    else:
        if not config['wandb']: # define a save_dir
            model_dir = "model_checkpoints/{}".format(config['run_name'])
            callbacks = [ModelCheckpoint(save_dir=model_dir, monitor=monitor)]
        else: # save to wandb.run.dir
            wandb_callback = CustomWandbCallback(config)
            model_dir = os.path.join(wandb_callback.wandb_run_dir, config['run_name'])
            callbacks = [ModelCheckpoint(save_dir=model_dir, monitor=monitor),
                        wandb_callback]
    config['model']['path'] = callbacks[0].save_path

    # Save the config for future reference
    config_save_fpath = os.path.join(model_dir, config['run_name']+'.yml')
    with open(config_save_fpath, 'w') as f:
        yaml.dump(config, f)

    # Compile and train
    model.compile(optimizer)
    print('Model Compiled.')
    history = model.fit(train_set,
                        validation_data=validation_set,
                        callbacks=callbacks,
                        epochs=train['epochs']) 