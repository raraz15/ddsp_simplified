from tensorflow.keras.callbacks import Callback
import wandb

WANDB_API_KEY = "52c84ab3f3b5c1f999c7f5f389f5e423f46fc04a"

class CustomWandbCallback(Callback):

    def __init__(self, run_name, **kwargs):
        super().__init__(**kwargs)

        wandb.login()
        wandb.init(project='Supervised_Violin', entity='ddsp', name=run_name)
        
    def on_epoch_end(self, epoch, logs=None):
        wandb.log(logs)
        
class ModelCheckpoint(Callback):
    def __init__(self, model, run_name, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.run_name = run_name

    def on_epoch_end(self, epoch, logs=None):
        self.model.save("{}/{}/model.ckpt".format(self.run_name, epoch))