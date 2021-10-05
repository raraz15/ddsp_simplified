import numpy as np
from dsp_utils import core
import tensorflow as tf

F0_RANGE = 127.0

def f0_scaled_L1_loss(x, x_pred):
    true = core.hz_to_midi(core.resample(x["f0_hz"], int(tf.shape(x_pred["inputs"]["f0_scaled"])[1])))/F0_RANGE
    pred = tf.squeeze(x_pred["inputs"]["f0_scaled"],axis=-1)
    return tf.math.reduce_mean(tf.math.abs(true-pred))