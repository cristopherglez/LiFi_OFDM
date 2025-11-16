import numpy as np
import scipy.io.wavfile as wav

def get_config():
    # -------- System parameters (must match TX) --------
    Lfft = 256  # FFT length
    cp_length = 16 
    oversampling_factor = 5  # Note: parameter name in constructor
    data_frame_length = 10
    lts_repetitions = 5
    sfo_repetitions = 4

    # -------- Load reference signals from WAV files --------

    sts_sample_rate, sts_signal = wav.read('/home/cris/Documents/Python/LiFi_OFDM/DEFzc_sequence_256_lfft_times5_44100ksps.wav')
    sts_no_cp = sts_signal.astype(np.int16)
    print(f"Loaded STS signal with length: {len(sts_no_cp)}")
    lts_sample_rate, lts_signal = wav.read('/home/cris/Documents/Python/LiFi_OFDM/DEFlts_256_lfft_times_5_44100ksps.wav')
    lts_no_cp = lts_signal.astype(np.int16)
    print(f"Loaded LTS signal with length: {len(lts_no_cp)}")

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
