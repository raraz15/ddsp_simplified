import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import metrics as tfkm

from synthesizers import *


class Autoencoder(Model):
    def __init__(self,
               preprocessor=None,
               add_reverb=False,
               loss_fn=None,
               n_samples=64000,
               sample_rate=16000,
               reverb_length=48000,
               tracker_names=["spec_loss"],
               metric_fns={},
               **kwargs):
        
        super().__init__(**kwargs)
        self.preprocessor = preprocessor
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.loss_fn = loss_fn
        
        self.harmonic = HarmonicSynthesizer(n_samples=self.n_samples, 
                                    sample_rate=self.sample_rate,
                                    name='harmonic')
        
        self.noise = FilteredNoiseSynthesizer(window_size=0,
                                      initial_bias=-10.0,
                                      name='noise')
        
        self.add_reverb = add_reverb
        if self.add_reverb:
            self.reverb = Reverb(reverb_length=reverb_length)
        self.trackers = TrackerGroup(*tracker_names)
        self.metric_fns = metric_fns
            
    def encode(self, features):
        raise NotImplementedError
    
    def decode(self, features):
        raise NotImplementedError
    
    def dsp_process(self, features):
        """Synthesizes audio and adds reverb if specified."""

        features["harmonic"] = self.harmonic(features) # synthesizes from f0_hz        
        features["noise"] = self.noise(features)
        outputs = {"inputs": features}
        outputs["audio_synth"] = features["harmonic"] + features["noise"]
        if self.add_reverb:
            outputs["audio_synth"] = self.reverb(outputs)            
        return outputs

    # code from github repo, kept it but unnecessary
    def get_audio_from_outputs(self, outputs):
        """Extract audio output tensor from outputs dict of call()."""
        return outputs['audio_synth']

    def transfer_timbre(self, features):
        model_output = self(features)
        audio_synth = self.get_audio_from_outputs(model_output)
        return audio_synth.numpy().reshape(-1)    

    def call(self, features):
        features = self.encode(features)
        features = self.decode(features)
        outputs = self.dsp_process(features)       
        return outputs        
  
    @tf.function
    def train_step(self, x):
        """Run the core of the network, get predictions and loss."""

        with tf.GradientTape() as tape:
            x_pred = self(x, training=True)
            loss = self.loss_fn({'audio': x_pred["audio_synth"] , 'target_audio':x["audio"]})
            total_loss = loss["total_loss"] if "total_loss" in loss else loss["spec_loss"]
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)      
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        metrics = {name:fn(x, x_pred) for name, fn in self.metric_fns.items()}
        self.trackers.update_state(loss)
        self.trackers.update_state(metrics)
        return self.trackers.result()
    
    @tf.function
    def test_step(self, x):
        x_pred = self(x,training=False)
        loss = self.loss_fn({'audio': x_pred["audio_synth"] , 'target_audio':x["audio"]})
        metrics = {name:fn(x, x_pred) for name, fn in self.metric_fns.items()}
        self.trackers.update_state(loss)
        self.trackers.update_state(metrics)
        return self.trackers.result()

    @property
    def metrics(self):
        return self.trackers.trackers.values()

class SupervisedAutoencoder(Autoencoder):
    def __init__(self,
               preprocessor=None,
               encoder=None,
               decoder=None,
               add_reverb=False,
               loss_fn=None,
               n_samples=64000,
               sample_rate=16000,
               reverb_length=48000,
               tracker_names=["spec_loss"],
               metric_fns={},
               **kwargs):
        
        super().__init__(preprocessor, add_reverb, loss_fn, n_samples, sample_rate, reverb_length=reverb_length,
                        tracker_names=tracker_names, metric_fns=metric_fns, **kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, features): 
        """Loudness and F0 is read. z is encoded optionally."""
        
        if self.preprocessor is not None: # Downsample and Scale the features
            processed_features = self.preprocessor(features)
            features.update(processed_features)
        if self.encoder is not None:
            outputs = self.encoder(features)
            features.update(outputs) 
        return features
    
    def decode(self, features):
        """Map the f,l (,z) parameters to synthesizer parameters."""
        
        outputs = self.decoder(features)       
        features.update(outputs)
        return features
     
class UnsupervisedAutoencoder(Autoencoder):
    def __init__(self,
               encoder,
               decoder,    
               preprocessor=None,
               add_reverb=False,
               loss_fn=None,
               n_samples=64000,
               sample_rate=16000,
               reverb_length=48000,
               tracker_names=["spec_loss"],
               metric_fns={},
               **kwargs):
        
        super().__init__(preprocessor, add_reverb, loss_fn, n_samples, sample_rate, reverb_length=reverb_length,
                         tracker_names=tracker_names, metric_fns=metric_fns, **kwargs)
        
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, features):
        if self.preprocessor is not None:
            features.update(self.preprocessor(features))
        outputs = self.encoder(features)
        features.update(outputs)
        return features
    
    def decode(self, features):  
        """Map the f, l (,z) parameters to synthesizer parameters."""
        output = self.decoder(features)   
        features.update(output)
        return features

class TrackerGroup():
    def __init__(self,*names):
        self.trackers = {name:tfkm.Mean(name+"_tracker") for name in names}

    def update_state(self, dct):
        for k,v in dct.items():
            self.trackers[k].update_state(v)
        
    def result(self):
        return {name:tracker.result() for name,tracker in self.trackers.items()}