from tensorflow.keras.callbacks import Callback
import wandb

class CustomWandbCallback(Callback):
    def __init__(self, run_name, **kwargs):
        super().__init__(**kwargs)
        wandb.init(project='DDSP', entity='haldunbalim33', name=run_name)
        
    def on_epoch_end(self, epoch, logs=None):
        wandb.log(logs)
        
class ModelCheckpoint(Callback):
    def __init__(self, model, run_name, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.run_name = run_name

    def on_epoch_end(self, epoch, logs=None):
        self.model.save("{}/{}/model.ckpt".format(self.run_name, epoch))
        
