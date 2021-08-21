import os, argparse, yaml

from dataloader import make_supervised_dataset
from train_utils import make_supervised_model, create_callbacks, make_optimizer

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Supervised Training Parameters.')
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to config file.')   
    args = parser.parse_args()

    # Read the config
    with open(args.path, 'r') as file:
        config = dict(yaml.load(file, Loader=yaml.FullLoader))
    
    # Create the datasets and the preprocessor
    train_set, validation_set, _ = make_supervised_dataset(config['data']['path'],
                                                    mfcc=config['model']['encoder'],
                                                    batch_size=config['training']['batch_size'],
                                                    sample_rate=config['data']['sample_rate'],
                                                    normalize=config['data']['normalize'],
                                                    conf_threshold=config['data']['confidence_threshold'])
    print('Dataset created.')
    
    # Create the model and define the training 
    model = make_supervised_model(config)
    optimizer = make_optimizer(config)

    # Model Saving and Experiment Tracking
    callbacks = create_callbacks(config, monitor='val_spec_loss')

    # Save the config for future reference
    config['model']['path'] = callbacks[0].save_path
    config_save_fpath = os.path.join(callbacks[0].save_dir, 'model.yaml')
    with open(config_save_fpath, 'w') as f:
        yaml.dump(config, f)

    # Compile and train
    model.compile(optimizer)
    print('Model compiled.')
    history = model.fit(train_set,
                        validation_data=validation_set,
                        callbacks=callbacks,
                        epochs=config['training']['epochs'])