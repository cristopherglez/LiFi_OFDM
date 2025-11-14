import numpy as np
import scipy.io.wavfile as wav

def get_config():
    # -------- System parameters (must match TX) --------
    Lfft = 256  # FFT length
    cp_length = 16 
    oversampling_factor = 5  # Note: parameter name in constructor
    data_frame_length = 10
    lts_repetitions = 5
    sfo_repetitions = 5

    # -------- Load reference signals from WAV files --------
    try:
        # Load STS from WAV file
        _, sts_signal = wav.read('FullReceiverSFO/DEFzc_sequence_256_lfft_times5_44100ksps.wav')
        sts_no_cp = sts_signal.astype(np.float32)
        print(f"Loaded STS signal with length: {len(sts_no_cp)}")
    except FileNotFoundError:
        print("STS WAV file not found, using placeholder")
        sts_no_cp = np.ones(Lfft, dtype=np.float32)
    
    try:
        # Load LTS from WAV file
        _, lts_signal = wav.read('/home/cris_/Documents/Python/FullReceiverSFO/DEFlts_256_lfft_times_5_44100ksps.wav')
        lts_no_cp = lts_signal.astype(np.float32)
        print(f"Loaded LTS signal with length: {len(lts_no_cp)}")
    except FileNotFoundError:
        print("LTS WAV file not found, using placeholder")
        lts_no_cp = np.ones(Lfft, dtype=np.float32)

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
