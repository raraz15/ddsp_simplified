import os
import numpy as np

from tensorflow.keras.callbacks import Callback

import wandb


class ModelCheckpoint(Callback):
    def __init__(self, save_dir, monitor, **kwargs):
        super().__init__(**kwargs)

        self.monitor = monitor
        
        self.save_dir = save_dir # wandb/run_name/files/run_name
        self.best_model_dir = os.path.join(save_dir, 'best_model')
        self.final_model_dir = os.path.join(save_dir, 'train_end')
        os.makedirs(self.best_model_dir, exist_ok=True)
        os.makedirs(self.final_model_dir, exist_ok=True)
        self.save_path = os.path.join(self.best_model_dir, 'model.ckpt')

        self.best = np.Inf
    
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if np.less(current, self.best):
            self.best = current
            self.model.save_weights(self.save_path)
            with open(os.path.join(self.best_model_dir, 'model_info.txt'), 'w') as outfile:
                outfile.write('epoch: {}'.format(epoch))
                for k, v in logs.items():
                    outfile.write('\n{}: {}'.format(k, v))

    # Save also the last model
    def on_train_end(self, logs=None):
        self.model.save_weights(os.path.join(self.final_model_dir, 'model.ckpt'))
        with open(os.path.join(self.final_model_dir, 'model_info.txt'), 'w') as outfile:   
            for k, v in logs.items():
                outfile.write('{}: {}\n'.format(k, v))              

class CustomWandbCallback(Callback):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        wandb.login()
        wandb.init(project=config['wandb']['project_name'],
                    entity='ddsp',
                    name=config['run_name'],
                    config=config)
        self.wandb_run_dir = wandb.run.dir
        
    def on_epoch_end(self, epoch, logs=None):
        wandb.log(logs)