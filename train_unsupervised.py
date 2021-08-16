import os, argparse, yaml

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ModelCheckpoint

from preprocessing import F0LoudnessPreprocessor # ?

from encoders import UnsupervisedEncoder
from models import UnsupervisedAutoencoder
from decoders import DecoderWithLatent
from losses import SpectralLoss, MultiLoss
from callbacks import CustomWandbCallback 

from dataloader import make_supervised_dataset # ??


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Supervised Training Parameters.')
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to config file.')   
    args = parser.parse_args()

    # Read the config
    with open(args.path) as file:
        config = dict(yaml.load(file, Loader=yaml.FullLoader))
    data, train = config['data'], config['training']
    model_config, optim = config['model'], config['optimizer']



    # ????????????*  
    # Create the datasets and the preprocessor
    train_set, validation_set, _ = make_supervised_dataset(data['path'],
                                                    model_config['encoder'],
                                                    train['batch_size'])
    print('Dataset created.')

    # ????????????*  
    preprocessor = F0LoudnessPreprocessor(timesteps=data['preprocessing_time'])


    # Create the model and define the training
    encoder = UnsupervisedEncoder()
    decoder = DecoderWithLatent(timesteps=model_config['decoder_time'])
    
    loss = SpectralLoss() if model_config['loss'] == 'spectral' else MultiLoss()
    if loss.name== 'spectral_loss':
        tracker_names = ['spec_loss']
        monitor = 'val_spec_loss'
    else:
        tracker_names = ['spec_loss', 'perc_loss', 'total_loss']
        monitor = 'val_total_loss'
    model = UnsupervisedAutoencoder(preprocessor=preprocessor,
                                encoder=encoder,
                                decoder=decoder,
                                loss_fn=loss,
                                tracker_names=tracker_names,
                                add_reverb=model_config['reverb'])
    optimizer = Adam(learning_rate=ExponentialDecay(optim['lr'],
                                decay_steps=optim['decay_steps'],
                                decay_rate=optim['decay_rate']))

    # TODO: use epoch name in model saving

    # Model Saving and Experiment Tracking
    if not config['wandb']:    
        if model_config['path']:
            model_path = model_config['path']
        else:
            model_path = "model_checkpoints/{}/model.ckpt".format(config['run_name'])
        callbacks = [ModelCheckpoint(filepath=model_path,
                                    monitor=monitor,
                                    save_best_only=True)]
    else:
        wandb_callback = CustomWandbCallback(config)
        if model_config['path']:
            model_path = model_config['path']
        else:
            model_dir = os.path.join(wandb_callback.wandb_run_dir, config['run_name'])
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir,
                                    "{}.ckpt".format(config['run_name']))
        callbacks = [ModelCheckpoint(filepath=model_path,
                                    monitor=monitor,
                                    save_best_only=True),
                    wandb_callback]

    # Save the config for future reference
    config_save_fpath = os.path.join(os.path.dirname(model_path), config['run_name']+'.yml')
    with open(config_save_fpath, 'w') as f:
        yaml.dump(config, f)

    # Compile and train
    model.compile(optimizer)
    print('Model Compiled.')
    history = model.fit(train_set,
                        validation_data=validation_set,
                        callbacks=callbacks,
                        epochs=train['epochs']) 