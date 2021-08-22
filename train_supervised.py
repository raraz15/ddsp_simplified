import os, argparse, yaml

from train_utils import make_supervised_model, create_callbacks, make_optimizer, make_supervised_dataset_from_config

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Supervised Training Parameters.')
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to the model config.')   
    args = parser.parse_args()

    # Read the config
    with open(args.path, 'r') as file:
        config = dict(yaml.load(file, Loader=yaml.FullLoader))
    
    # Create the datasets
    train_set, validation_set, _ = make_supervised_dataset_from_config(config)
    print('Dataset created.')
    
    # Create the entire model and define the training 
    model = make_supervised_model(config)
    optimizer = make_optimizer(config)

    # Plan the Model Saving and Experiment Tracking
    callbacks = create_callbacks(config, monitor='val_spec_loss')

    # Save the config for future reference
    config['model']['path'] = callbacks[0].save_path
    with open(os.path.join(callbacks[0].save_dir, 'model.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Compile and train
    model.compile(optimizer)
    print('Model compiled.')
    history = model.fit(train_set,
                        validation_data=validation_set,
                        callbacks=callbacks,
                        epochs=config['training']['epochs'])