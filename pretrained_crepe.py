import tensorflow as tf
from tensorflow.keras import layers as tfkl
import crepe
from dsp_utils import spectral_ops

# USED IN F0 EXTRACTION DURING TIMBRE TRANSFER OF THE SECOND SOURCE
class PretrainedCREPE(tfkl.Layer):
    """Pretrained CREPE model with frozen weights."""

    def __init__(self,
               model_capacity='tiny',
               activation_layer='conv5-maxpool',
               name='pretrained_crepe',
               trainable=False):
        super().__init__(name=name, trainable=trainable)
        self._model_capacity = model_capacity
        self._activation_layer = activation_layer
        spectral_ops.reset_crepe()
        self._model = crepe.core.build_and_load_model(self._model_capacity)
        self.frame_length = 1024

    def build(self, unused_x_shape):
        self.layer_names = [l.name for l in self._model.layers]

        if self._activation_layer not in self.layer_names:
            raise ValueError(
              'activation layer {} not found, valid names are {}'.format(
                  self._activation_layer, self.layer_names))

        self._activation_model = tf.keras.Model(
            inputs=self._model.input,
            outputs=self._model.get_layer(self._activation_layer).output)

        # Variables are not to be trained.
        self._model.trainable = self.trainable
        self._activation_model.trainable = self.trainable

    def frame_audio(self, audio, hop_length=1024, center=True):
        """Slice audio into frames for crepe."""
        # Pad so that frames are centered around their timestamps.
        # (i.e. first frame is zero centered).
        pad = int(self.frame_length / 2)
        audio = tf.pad(audio, ((0, 0), (pad, pad))) if center else audio
        frames = tf.signal.frame(audio,
                                 frame_length=self.frame_length,
                                 frame_step=hop_length)

        # Normalize each frame -- this is expected by the model.
        mean, var = tf.nn.moments(frames, [-1], keepdims=True)
        frames -= mean
        frames /= (var**0.5 + 1e-5)
        return frames

    def call(self, audio):
        """Returns the embeddings.

        Args:
          audio: tensors of shape [batch, length]. Length must be divisible by 1024.

        Returns:
          activations of shape [batch, depth]
        """
        frames = self.frame_audio(audio)
        batch_size = int(tf.shape(frames)[0])
        n_frames = int(tf.shape(frames)[1])
        # Get model predictions.
        frames = tf.reshape(frames, [-1, self.frame_length])
        outputs = self._activation_model(frames)
        outputs = tf.reshape(outputs, [batch_size, n_frames, -1])
        return outputs