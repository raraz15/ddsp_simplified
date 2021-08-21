import numpy as np

from dsp_utils.spectral_ops import compute_loudness, compute_f0, compute_mfcc, compute_logmel

from utilities import concat_dct, frame_generator

## -------------------------------------------------- Feature Extraction ---------------------------------------

# TODO: decide on nffts
def feature_extractor(audio_frame, sample_rate=16000, frame_rate=250,
                    f0=True, loudness=True, mfcc=False, log_mel=False,
                    l_nfft=2048,mfcc_nfft=1024, logmel_nfft=2048,
                    model=None, conf_threshold=0.0):
    """Extracts features for a single frame."""
    
    features = {'audio': audio_frame}

    if f0:
        f0, confidence = compute_f0(audio_frame, sample_rate, frame_rate, viterbi=True) 
        f0 = confidence_filter(f0, confidence, conf_threshold)
        features['f0_hz'] = f0

    if mfcc:
        # overlap and fft_size taken from the code
        features['mfcc'] = compute_mfcc(audio_frame,
                                        lo_hz=20.0,
                                        hi_hz=8000.0,
                                        fft_size=mfcc_nfft,
                                        overlap=0.75,
                                        mel_bins=128,
                                        mfcc_bins=30)

    if log_mel:
        features['log_mel'] = compute_logmel(audio_frame,
                                            lo_hz=80.0,
                                            hi_hz=7600.0,
                                            bins=64,
                                            fft_size=logmel_nfft,
                                            overlap=0.75,
                                            pad_end=True,
                                            sample_rate=sample_rate)
                                                                             
    if loudness:                                            
        # apply reverb before l extraction to match room acoustics
        # used during timbre transfer
        if model is not None and model.add_reverb: 
            audio_frame = model.reverb({"audio_synth":audio_frame[np.newaxis,:]})[0]

        features['loudness_db'] = compute_loudness(audio_frame,
                                                    sample_rate=sample_rate,
                                                    frame_rate=frame_rate,
                                                    n_fft=l_nfft,
                                                    range_db=120.0,
                                                    ref_db=20.7,
                                                    use_tf=False)                             
    return features

def extract_features_from_frames(frames, **kwargs):
    """Extracts features from multiple frames and concatenates them."""
    return concat_dct([feature_extractor(frame, **kwargs) for frame in frames])    

def process_track(track, sample_rate=16000, audio_length=60, frame_size=64000, **kwargs):
    """Generates frames from a track and extracts features for each frame."""
    MAX_AUDIO_LENGTH = sample_rate*audio_length
    if len(track) > MAX_AUDIO_LENGTH: # trim from the end
        track = track[:MAX_AUDIO_LENGTH]
    frames = frame_generator(track, frame_size) # large chunks of audio
    return extract_features_from_frames(frames, **kwargs)

def confidence_filter(F0, confidence, threshold):
    """
    Silences the time instants where the model confidence is below the given threshold.
    """

    return [f if confidence[idx] >= threshold else 0.0 for idx, f in enumerate(F0)]    