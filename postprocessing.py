import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, convolve, get_window
import librosa

# EXcept the plots mostly unused

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