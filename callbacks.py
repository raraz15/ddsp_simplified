import os
import numpy as np

from tensorflow.keras.callbacks import Callback

import wandb



class ModelCheckpoint(Callback):
    def __init__(self, save_dir, monitor, save_every=250, **kwargs):
        super().__init__(**kwargs)

        self.monitor = monitor
        
        self.save_dir = save_dir # wandb/run_name/files/run_name

        self.best_model_dir = os.path.join(save_dir, 'best_model')
        self.final_model_dir = os.path.join(save_dir, 'train_end')
        self.save_every_dir = os.path.join(save_dir, 'save_every')
        for d in [self.best_model_dir, self.final_model_dir, self.save_every_dir]:
            os.makedirs(d, exist_ok=True)
        
        self.save_path = os.path.join(self.best_model_dir, 'model.ckpt') # for loading the config back

        self.best = np.Inf

        self.save_every = save_every
        self.epoch_counter = 0
    
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if np.less(current, self.best):
            self.best = current
            self.save_weights(logs, self.best_model_dir, epoch)

        self.epoch_counter +=1
        if self.epoch_counter == self.save_every:
            save_dir = os.path.join(self.save_every_dir, str(epoch))
            os.makedirs(save_dir, exist_ok=True)
            self.save_weights(logs, save_dir, epoch)           
            self.epoch_counter = 0
            
    # Save also the last model
    def on_train_end(self, logs=None):
        self.save_weights(self.final_model_dir, logs)

    def save_weights(self, logs, dir, epoch=None):
        self.model.save_weights(os.path.join(dir, 'model.ckpt'))
        with open(os.path.join(dir, 'model_info.txt'), 'w') as outfile:
            if epoch is not None: 
                outfile.write('epoch: {}\n'.format(epoch))  
            for k, v in logs.items():
                outfile.write('{}: {}\n'.format(k, v))


class CustomWandbCallback(Callback):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        wandb.login()
        wandb.init(project=config['wandb']['project_name'],
                    entity=config['wandb']['entity'],
                    name=config['run_name'],
                    config=config)
        self.wandb_run_dir = wandb.run.dir
        
    def on_epoch_end(self, epoch, logs=None):
        wandb.log(logs)