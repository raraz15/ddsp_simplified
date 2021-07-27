import tensorflow as tf
from tensorflow.keras import layers as tfkl
from dsp_utils import spectral_ops

from pretrained_crepe import PretrainedCREPE


class SpectralLoss(tfkl.Layer):

    def __init__(self,
                 fft_sizes=(2048, 1024, 512, 256, 128, 64),
                 logmag_weight=1.0,
                 name='spectral_loss'):

        super().__init__(name=name)
        self.fft_sizes = fft_sizes     
        self.logmag_weight = logmag_weight

    def call(self, inputs):
        audio, target_audio = inputs["audio"], inputs["target_audio"]
        total_loss = 0
        for size in self.fft_sizes:
            total_loss += self.calculate_loss_for_fft_size(audio,target_audio,size)  
        return {"spec_loss": total_loss}
    
    def calculate_loss_for_fft_size(self, audio, target_audio, size):
        # magnitude spectrograms of the true audio and the sythesized version
        mag_audio = spectral_ops.compute_mag(audio,size=size) 
        mag_target = spectral_ops.compute_mag(target_audio,size=size)
        log_mag_audio  = spectral_ops.safe_log(mag_audio)
        log_mag_target_audio = spectral_ops.safe_log(mag_target)
        mag_loss = tf.math.reduce_mean(tf.math.abs(mag_audio - mag_target))
        log_mag_loss = tf.math.reduce_mean(tf.math.abs(log_mag_audio - log_mag_target_audio))  
        return mag_loss + self.logmag_weight * log_mag_loss
    

class PerceptualLoss(tfkl.Layer):

    def __init__(self,
               weight=38.0,
               model_capacity='tiny',
               name='pretrained_crepe_embedding_loss',
               activation_layer='conv5-BN'):
        
        super().__init__(name=name)
        self.weight = weight
        self.pretrained_model = PretrainedCREPE(model_capacity=model_capacity,
                                             activation_layer=activation_layer)

    def call(self, inputs):
        
        audio, target_audio = inputs["audio"], inputs["target_audio"]
        audio, target_audio = tf_float32(audio), tf_float32(target_audio)
        
        target_emb = self.pretrained_model(target_audio)
        synth_emb = self.pretrained_model(audio)
        
        loss = self.weight * tf.reduce_mean(tf.abs(target_emb - synth_emb))
        
        return {"perc_loss":loss}

    
class MultiLoss(tfkl.Layer):
    
    def __init__(self, logmag_weight=1.0, perceptual_loss_weight=5e-5, name="multi_loss"):
        super().__init__(name=name)
        self.spec_loss_fn = SpectralLoss(logmag_weight=logmag_weight)
        self.perceptual_loss_fn = PerceptualLoss(weight=perceptual_loss_weight)
    
    def call(self, inputs):
        spec_loss = self.spec_loss_fn(inputs)['spec_loss']
        perc_loss = self.perceptual_loss_fn(inputs)['perc_loss']
        total_loss = spec_loss+perc_loss
        return {"total_loss":total_loss, "spec_loss": spec_loss, "perc_loss": perc_loss}
    
    
def tf_float32(x):
    return tf.cast(x,dtype=tf.float32)