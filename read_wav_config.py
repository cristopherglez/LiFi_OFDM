import os
import numpy as np
import scipy.io.wavfile as wav

def get_wav_config():
    # -------- System parameters (must match TX) --------
    Lfft = 256  # FFT length
    cp_length = 16 
    oversampling_factor = 5
    data_frame_length = 10
    lts_repetitions = 5
    sfo_repetitions = 3

    # -------- Load reference signals from WAV files --------
    base_dir = os.path.dirname(__file__)
    
    # Load STS reference signal
    sts_path = os.path.join(base_dir, 'DEFzc_sequence_256_lfft_times5_44100ksps_4_symbols_v2.wav')
    #sts_path = os.path.join(base_dir, 'DEFzc_sequence_256_lfft_times5_44100ksps.wav')
    sts_sample_rate, sts_signal = wav.read(sts_path)
    if sts_signal.ndim > 1:
        sts_no_cp = sts_signal[:, 0].astype(np.int16)
    else:
        sts_no_cp = sts_signal.flatten().astype(np.int16)
    print(f"Loaded STS signal with shape: {sts_no_cp.shape}")
    
    # Load LTS reference signal
    lts_path = os.path.join(base_dir, 'DEFlts_256_lfft_times_5_44100ksps_4_symbols_v2.wav')
    #lts_path = os.path.join(base_dir, 'DEFlts_256_lfft_times_5_44100ksps.wav')
    
    lts_sample_rate, lts_signal = wav.read(lts_path)
    if lts_signal.ndim > 1:
        lts_no_cp = lts_signal[:, 0].astype(np.int16)
    else:
        lts_no_cp = lts_signal.flatten().astype(np.int16)
    print(f"Loaded LTS signal with shape: {lts_no_cp.shape}")
    
    # Load the full TX WAV file (recorded transmission)
    input_wav_path = os.path.join(base_dir, 'DEFfull_tx256_lfft_times5_44100ksps_definitiva_rand.wav')
    #input_wav_path = os.path.join(base_dir, 'DEFfull_tx256_lfft_times5_44100ksps.wav')
    input_sample_rate, input_signal = wav.read(input_wav_path)
    if input_signal.ndim > 1:
        input_signal = input_signal[:, 0].astype(np.float32)
    else:
        input_signal = input_signal.flatten().astype(np.float32)
    print(f"Loaded input WAV signal with shape: {input_signal.shape}, sample rate: {input_sample_rate} Hz")

    # Add random padding between 1 and Lfft*oversampling_factor samples
    padding_length = np.random.randint(1, Lfft * oversampling_factor + 1)
    padding = np.random.randn(padding_length).astype(np.float32)
    input_signal = np.concatenate((padding, input_signal, padding))

    return {
        'Lfft': Lfft,
        'cp_length': cp_length,
        'data_frame_length': data_frame_length,
        'lts_repetitions': lts_repetitions,
        'sfo_repetitions': sfo_repetitions,
        'sts_no_cp': sts_no_cp,
        'lts_no_cp': lts_no_cp,
        'oversampling_factor': oversampling_factor,
        'input_signal': input_signal,
        'sample_rate': input_sample_rate,
    }
