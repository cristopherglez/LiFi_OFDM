import os
import numpy as np
import scipy.io.wavfile as wav

def get_config():
    # -------- System parameters (must match TX) --------
    Lfft = 128  # FFT length
    cp_length = int(Lfft / 8)  # Cyclic prefix length
    oversampling_factor = 10  # Note: parameter name in constructor
    data_frame_length = 5
    lts_repetitions = 5
    sfo_repetitions = 3

    # -------- Load reference signals from WAV files --------

    # Load WAV files relative to this config file so the repo can be used on any OS
    base_dir = os.path.dirname(__file__)
    sts_path = os.path.join(base_dir, 'DEFzc_sequence_256_lfft_times5_44100ksps.wav')
    sts_sample_rate, sts_signal = wav.read(sts_path)
    # Ensure 1D array - if stereo, take first channel; if mono, flatten
    if sts_signal.ndim > 1:
        sts_no_cp = sts_signal[:, 0].astype(np.int16)  # Take first channel
    else:
        sts_no_cp = sts_signal.flatten().astype(np.int16)  # Ensure 1D
    print(f"Loaded STS signal with shape: {sts_no_cp.shape}")
    
    lts_path = os.path.join(base_dir, 'DEFlts_256_lfft_times_5_44100ksps.wav')
    lts_sample_rate, lts_signal = wav.read(lts_path)
    # Ensure 1D array - if stereo, take first channel; if mono, flatten  
    if lts_signal.ndim > 1:
        lts_no_cp = lts_signal[:, 0].astype(np.int16)  # Take first channel
    else:
        lts_no_cp = lts_signal.flatten().astype(np.int16)  # Ensure 1D
    print(f"Loaded LTS signal with shape: {lts_no_cp.shape}")

    return {
        'Lfft': Lfft,
        'cp_length': cp_length,
        'data_frame_length': data_frame_length,
        'lts_repetitions': lts_repetitions,
        'sfo_repetitions': sfo_repetitions,
        'sts_no_cp': sts_no_cp,
        'lts_no_cp': lts_no_cp,
        'oversampling_factor': oversampling_factor,
    }
