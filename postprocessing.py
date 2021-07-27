import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, convolve, get_window
from scipy.io.wavfile import write
import librosa

from dsp_utils.spectral_ops import compute_loudness, compute_f0, compute_mfcc

## -------------------------------------------------- Feature Extraction ---------------------------------------

# TODO: mfcc integration
def process_track(track, mfcc=False, loudness_nfft=2048, frame_size=64000, Fs=16000, frame_rate=250, pitch_shift=0, model=None, normalize=True):
    """Generates frames and extracts l, f0 for each frame."""

    MAX_AUDIO_LENGTH = Fs*32
    if len(track) > MAX_AUDIO_LENGTH: # trim from the end
        track = track[:MAX_AUDIO_LENGTH]
    if pitch_shift:
        track = librosa.effects.pitch_shift(track, Fs, n_steps=pitch_shift)
    if normalize:
        track = librosa.util.normalize(track)
    audio_frames = frame_generator(track, frame_size) # large chunks of audio

    features = np.array([extract_features(frame, mfcc, loudness_nfft, Fs, frame_rate, model) for frame in audio_frames])

    if mfcc:
        return {'audio': audio_frames,
            'loudness_db': features[:,0,:],
            'f0_hz': features[:,1,:],
            'mfcc': features[:,2,:]}
    else:
        return {'audio': audio_frames,
            'loudness_db': features[:,0,:],
            'f0_hz': features[:,1,:]}        

def extract_features(audio, mfcc=False, loudness_nfft=2048, Fs=16000, frame_rate=250, model=None):
    f0, _ = compute_f0(audio, Fs, frame_rate, viterbi=True)
    if mfcc:
        m = compute_mfcc(audio, lo_hz=20.0, hi_hz=8000.0,
                        fft_size=1024, mel_bins=128, mfcc_bins=30)
    if model is not None and model.add_reverb==False: # apply reverb before l extraction to match room acoustics
        audio = model.reverb({"audio_synth":audio[np.newaxis,:]})[0]
    l = compute_loudness(audio,
                    sample_rate=Fs,
                    frame_rate=frame_rate,
                    n_fft=loudness_nfft,
                    range_db=120.0,
                    ref_db=20.7,
                    use_tf=False)
    if mfcc:
        return l, f0, m
    else:
        return l, f0


def frame_generator(track, frame_size=64000):
    return track[:len(track)-len(track)%frame_size].reshape(-1,frame_size)

## -------------------------------------------- Timbre Transfer -------------------------------------------------

def transfer_timbre(model, features):
    model_output = model(features)
    audio_synth = model_output['audio_synth']
    return audio_synth.numpy().reshape(-1)

# scale loudness ?
def transfer_timbre_from_filepath(model, path, Fs=16000, scale_loudness=0, **kwargs):
    track, _ = librosa.load(path,sr=Fs)
    features = process_track(track, model=model, **kwargs)
    features["loudness_db"] +=  scale_loudness
    return transfer_timbre(model,features)

def write_audio(audio, title, RUN_NAME, Fs=16000, normalize=True):
    assert '.wav' in title, 'Title must include .wav extension'
    output_path = os.path.join('audio_clips','outputs', RUN_NAME)
    os.makedirs(output_path, exist_ok=True)
    if normalize:
        audio = librosa.util.normalize(audio)
    write(os.path.join(output_path,title), Fs, audio)

# -------------------------------------------------- Plots -----------------------------------------------------

def extract_dB_spectrogram(audio, n_fft, win_length, hop_length, center=True):

    assert win_length < n_fft, 'Window length must be greater than N_fft'
    
    amplitude_spectrogram = np.abs(librosa.stft(audio, 
                                        n_fft=n_fft, 
                                        win_length=win_length, 
                                        hop_length=hop_length,
                                        center=center))
                                           
    return librosa.amplitude_to_db(amplitude_spectrogram, np.max(amplitude_spectrogram))

def plot_spectrogram(track, sr=16000, n_fft=4096, win_length=1024, hop_length=512, center=True):
    
    dB_spectrogram = extract_dB_spectrogram(track, n_fft, win_length, hop_length, center=True)

    fig, ax = plt.subplots(figsize=(20,8))
    
    librosa.display.specshow(dB_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    
    plt.show()
    

def plot_waveform_spectrogram(track, sr=16000, n_fft=4096, win_length=1024, hop_length=512, center=True, title=''):
    
    dB_spectrogram = extract_dB_spectrogram(track, n_fft, win_length, hop_length, center=True)

    fig, ax = plt.subplots(figsize=(20,8), nrows=2, sharex=True, constrained_layout=True) #, dpi=50
    
    fig.suptitle('Synthesized Violin, Timber Transferred from Singing Voice', fontsize=15)
    ax[0].set_title('Spectrogram',fontsize=13)
    ax[1].set_title('Waveform', fontsize=13)

    librosa.display.specshow(dB_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax[0])
    librosa.display.waveplot(track, sr=sr, ax=ax[1])
    
    plt.savefig(title)
    plt.show()

## -----------------------------------------

def generate_windowed_frames(x, frame_size, window_type):

    audio_frames = frame_generator(x, frame_size=frame_size)

    window = get_window(window_type, frame_size, fftbins=False)

    windowed_frames = [frame*window for frame in audio_frames]
    
    return windowed_frames

def reconstruct(windowed_frames, frame_size):
    """
    Overlap-add method with 50% overlap
    """
    
    reconstruction = [windowed_frames[0][:frame_size//2]] # first frame's beginning
    for i in range(len(windowed_frames)-1):

         reconstruction += [windowed_frames[i][frame_size//2:] + windowed_frames[i+1][:frame_size//2]]

    reconstruction += [windowed_frames[i][frame_size//2:]] # last frames end

    reconstruction = np.array(reconstruction).reshape(-1)
    
    return reconstruction


def lp_and_normalize(track, fc, fs, M=5001, window_type='blackman'):
    """
    Low Pass filters the track with same length convolution and normalizes the output.
    Causal, Generalized Phase Type I filter.

        Parameters:
        -----------
            track(ndarray): audio track
            fc (float): Cut-off frequency in Hz
            fs (float): Sampling frequency in Hz
            N (int, default=5001): Filter tap
            window_type (str, default='blackmann'): window type
        
        Returns:
        --------
            track_cut (ndarray): processed track
    """

    # Type I filter
    lp = firwin(M,
                cutoff=fc,
                window=window_type,
                fs=fs) 

    track_cut = convolve(track, lp, mode='same') # same length convolution

    track_cut = normalize(track_cut) # normalize

    return track_cut