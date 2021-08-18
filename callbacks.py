import os
import numpy as np

from tensorflow.keras.callbacks import Callback

import wandb

WANDB_API_KEY = "52c84ab3f3b5c1f999c7f5f389f5e423f46fc04a"


class ModelCheckpoint(Callback):
    def __init__(self, save_dir, monitor, **kwargs):
        super().__init__(**kwargs)

        self.monitor = monitor
        
        self.save_dir = save_dir  
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, 'model.ckpt')

        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if np.less(current, self.best):
            self.best = current
            self.model.save_weights(self.save_path)
            with open(os.path.join(self.save_dir, 'model_info.txt'), 'w') as outfile:
                outfile.write('epoch: {}, {}: {}'.format(epoch, self.monitor, current))


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