import numpy as np
from scipy.signal import correlate

class OFDMReceiver:
    def __init__(self, Lfft, cp_length, data_frame_length, lts_repetitions,
                 sts_no_cp, lts_no_cp, oversampling_factor=1):
        # System
        self.oversampling_factor = oversampling_factor
        self.Lfft = Lfft
        self.Nsub = Lfft // 2 - 1  
        self.cp_length = cp_length
        self.full_symbol_length = Lfft + cp_length
        self.window_length = self.full_symbol_length * oversampling_factor
        self.data_frame_length = data_frame_length
        self.lts_repetitions = lts_repetitions

        # References
        self.sts_no_cp = sts_no_cp
        self.lts_no_cp = lts_no_cp

        # QPSK reference
        self.qpsk_points = np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype=complex)

        # State
        self.start_flag = False
        self.start_index = 0
        self.i = 0
        self.sto_acc=0.0
        self.Eq = np.zeros(self.Nsub, dtype=complex)
        self.y = np.array([], dtype=complex)
        self.sto_correction = -int((cp_length/2 - 1)*oversampling_factor)
        self.sto_counter = 0.0

    def packet_detection(self, received_signal):
        """Detect packet start using cross-correlation with STS.
        Args:
            received_signal (np.ndarray): The received signal array.
        Returns:
            start_flag (bool): True if packet is detected, False otherwise.
            start_index (int): The index of the detected packet start.
        """
        # Normalize STS and received signal
        sts_norm = self.sts_no_cp / np.max(np.abs(self.sts_no_cp))
        signal_norm = received_signal / np.max(np.abs(received_signal))

        # Compute auto-correlation of known STS
        sts_auto_corr = correlate(sts_norm, sts_norm, mode='full')
        threshold = 0.25 * np.max(np.abs(sts_auto_corr))

        # Compute normalized cross-correlation
        correlation_values = correlate(signal_norm, sts_norm, mode='valid')
        peak_value = np.max(np.abs(correlation_values))
        start_index = np.argmax(np.abs(correlation_values))
        
        if start_index < self.window_length:
            start_flag = peak_value > threshold
        else:
            start_flag = False
        # Notify if packet is detected
        if start_flag:
            print(f"Packet detected at index: {start_index}")
        return start_flag, start_index, correlation_values, sts_auto_corr    

    def channel_estimation_ls(self, received_symbol_no_cp):
        #received_symbol_no_cp = np.real(received_symbol_no_cp)/np.max(np.abs(received_symbol_no_cp))
        # #print lengths
        ##print(f"Length of LTS (no CP): {len(self.lts_no_cp)}")
        ##print(f"Length of received symbol (no CP): {len(received_symbol_no_cp)}")
        #delta = np.zeros(self.Lfft * self.oversampling_factor)
        #delta[1] = 1
        ##print(f"Delta length: {len(delta)}")
        #data = np.ones_like(self.Nsub, dtype=complex)*(1 + 1j)
        data = np.zeros(self.Nsub+1, dtype=complex)
        """data[:self.Nsub//4] = 1 + 1j
        data[self.Nsub//4:self.Nsub//2] = -1 + 1j
        data[self.Nsub//2:3*self.Nsub//4] = -1 -1j    
        data[3*self.Nsub//4:] = 1 - 1j
        data = np.concatenate((np.zeros(1, dtype=complex), data))
        data_ask = np.conj(np.flip(data))
        new_data = np.concatenate((data, data_ask))
        #print(f"Data length: {len(new_data)}")"""
        #data[1:] = 1 + 0j
        zc_data = self.generate_complex_zc(61)  # Scale to match QPSK energy
        data[1:] = zc_data
        #X = np.fft.fft(np.real(self.lts_no_cp), n=self.Lfft)[1:self.Nsub+1]
        #X = np.fft.fft(delta, n=self.Lfft)[1:self.Lfft // 2]
        spectrum = np.zeros(self.Nsub + 1, dtype=complex)
        spectrum[1:] = self.recover_dco_ofdm(received_symbol_no_cp)
        Eq = data[1:]/spectrum[1:]
        #Eq = spectrum[1:]/data[1:]
        ##print(f"Eq length: {len(Eq)}")
        return Eq
        
    def generate_complex_zc(self, u: int):
        """
        Genera una secuencia de Zadoff-Chu con índice raíz u y longitud Nzc, repetida.
        
        Entradas:
        - u: Índice raíz de la secuencia Zadoff-Chu.
        - Nzc: Longitud de la secuencia Zadoff-Chu.
        Salida:
        - zc_sequence: Secuencia Zadoff-Chu compleja.
        """
        n = np.arange(self.Nsub)
        zc_sequence = np.exp(-1j * np.pi * u * n * (n + 1) / self.Nsub)
        return zc_sequence

    def recover_dco_ofdm(self, input_symbol_no_cp):
        input = []
        for i in range(0, self.Lfft):
            new_index = int(i*self.oversampling_factor)
            input.append(input_symbol_no_cp[new_index])
        #print(f"Decimated input length: {len(input)}")
        data = np.fft.fft(input)[1:self.Nsub+1]
        return data
    
    def qpsk_demod(self, symbols):
        if symbols.size == 0:
            return np.array([], dtype=np.uint8)
        # Optional normalization toward unit energy
        s = symbols / (np.sqrt(2) + 1e-12)
        bits = np.empty(2 * s.size, dtype=np.uint8)
        # Map signs to bits according to TX mapping:
        # (0,0)->1+1j, (0,1)->1-1j, (1,0)->-1+1j, (1,1)->-1-1j
        r_nonneg = (np.real(s) >= 0).astype(np.uint8)
        i_nonneg = (np.imag(s) >= 0).astype(np.uint8)
        # We defined re>=0 -> bit 0, re<0 -> bit 1; same for imag
        # Place as [b0, b1, b0, b1, ...]
        bits[0::2] = 1 - r_nonneg  # re>=0 => 0; re<0 => 1
        bits[1::2] = 1 - i_nonneg  # im>=0 => 0; im<0 => 1
        return bits

# Unused function
    def minn_method_sto_estimation(self, received_signal):
        corr_length = self.window_length + (self.cp_length * self.oversampling_factor) - 1
        minn_metric = np.zeros(corr_length, dtype=complex)
        P = np.zeros(corr_length, dtype=complex)
        R = np.zeros(corr_length, dtype=complex)
        L = self.Lfft*self.oversampling_factor//4
        for d in range(corr_length - 1):
            a_1 = received_signal[d:d + L - 1]
            a_2 = received_signal[d + L: d + 2*L -1]
            a_3 = received_signal[d + 2*L: d + 3*L -1]
            a_4 = received_signal[d + 3*L: d + 4*L -1]
            b_1 = np.abs(received_signal[d + L: d + 2*L -1])**2
            b_2 = np.abs(received_signal[d + 3*L: d + 4*L -1])**2
            if len(a_4) != len(a_1):
                print(f"Length mismatch at index {d}: len(a_1)={len(a_1)}, len(a_4)={len(a_4)}")
                print(f"Signal full length: {len(received_signal)}, d: {d}, L: {L}")
            p = np.sum(np.vdot(a_1, a_2) + np.vdot(a_3, a_4))
            r = np.sum(b_1 + b_2)
            if len(a_1) < self.cp_length or len(a_2) < self.cp_length:
                print(f"Insufficient length for Minn's correlation at index {d}")
                minn_metric[d] = 0
                continue
            P[d] = abs(p)**2
            R[d] = r**2
            minn_metric[d] = P[d]*(R[d])
        sto_index = int(np.argmax(np.abs(minn_metric)))
        return sto_index, np.sum(minn_metric), minn_metric
    
    def index_correction(self):
        self.sto_acc += self.sto_frac
        if self.sto_acc >= 1.0:
            self.sto_frac_corr = 1
            self.sto_acc -= 1.0
        else:
            self.sto_frac_corr = 0
        if self.start_index + self.sto_int + self.sto_frac_corr < 0:
            self.start_index += int(self.sto_int) + self.sto_frac_corr + self.window_length
            print("ERROR: Start index negative after SFO correction")
            #print(f"New start index: {self.start_index}")
        else: 
            self.start_index += int(self.sto_int) + int(self.sto_frac_corr)
        pass

# Unused function
    def interpolate_correction(self, signal):
        real_length = (self.Lfft * self.oversampling_factor)
        length_with_sfo =  real_length + self.sto_correction
        end_idx = self.start_index + (self.Lfft*self.oversampling_factor) + self.sto_int
        chunk = signal[self.start_index : end_idx]
        chunk = np.interp(np.linspace(0, len(chunk), int(real_length), endpoint=False),
                              np.arange(len(chunk)), chunk)
        return chunk

    def process_zc(self, x1, x2):
        signal = np.concatenate([x1, x2]) 
        if not self.start_flag: 
            # Perform packet detection, coarse sync
            self.start_flag, self.start_index, _, _ = self.packet_detection(signal)
        else: 
            if self.i >= 0 and self.i < (self.lts_repetitions): 
                # Periodic STO estimation (SFO estimation)
                old_idx = self.start_index
                if self.i >= 1:
                    self.start_index = np.argmax(np.absolute(correlate(signal, self.sts_no_cp, mode='valid')[:self.Lfft*self.oversampling_factor])) 
                self.sto_correction += old_idx - self.start_index
                chunk = signal[self.start_index : self.start_index + self.Lfft*self.oversampling_factor]
                if self.i > 0:
                    #print(f"Adjusted start index for LTS processing: {self.start_index}")
                    self.sto_counter += self.start_index - old_idx
                    #print(f"Offset applied to start index: {self.start_index - old_idx}")
                if self.i == (self.lts_repetitions - 1):
                    self.sto_correction = (self.sto_counter / (self.lts_repetitions))
                    #self.sto_correction = -1.25
                    #print(f"Samples offset per buffer: {self.sto_correction}")
                    self.sto_int = int(self.sto_correction)
                    #print(f"Integer STO correction (samples): {self.sto_int}")
                    self.sto_frac = self.sto_correction - self.sto_int
                    #print(f"Fractional STO correction (samples): {self.sto_frac}")
                    self.sto_frac_corr = 0 # Initialize fractional correction accumulator
                    #print(f"Final STO correction after LTS processing: {self.sto_int + self.sto_frac}")
                if self.i > 0:
                    # Channel response estimation
                    self.Eq += self.channel_estimation_ls(chunk)
                # Finalize LTS estimation
                self.Eq /= self.lts_repetitions - 1
                #self.Eq /= self.generate_complex_zc(61) # TEMPORARY SET TO ZC sequence FOR TESTING
                #self.Eq = np.ones(self.Nsub, dtype=complex)  # TEMPORARY SET TO ONES FOR TESTING
                #print(f"Final channel equalizer Eq computed.")
                self.y = correlate(signal, self.sts_no_cp, mode='valid')
                self.i += 1
                return self.start_flag, self.start_index, self.y, self.i, self.Eq
            elif(self.i >= self.lts_repetitions) and (self.i < self.data_frame_length + self.lts_repetitions):
                # Data frame processing    
                self.index_correction() # STO correction
                chunk = signal[self.start_index - 40: self.start_index + self.Lfft * self.oversampling_factor - 40]
                # Process data frames
                self.y = self.recover_dco_ofdm(chunk) * self.Eq
                pilot_tones_indexes = [0, 13, 25, 38, 50, 62]
                # Check pilot tones
                pilot_values = self.y[pilot_tones_indexes]
                angles = np.angle(pilot_values)
                unwrapped_angles = np.unwrap(angles)
                display_vector = np.zeros_like(self.y)
                # Interpolate to find the phase correction for all subcarriers
                display_vector = np.interp(np.arange(len(self.y)), pilot_tones_indexes, unwrapped_angles)
                #Now correct the symbols with the interpolated phase correction
                self.y = self.y * np.exp(-1j * display_vector)
                self.i += 1
                return self.start_flag, self.start_index, self.y, self.i, self.Eq
            else:
                if self.i == self.data_frame_length + self.lts_repetitions + 1:
                    print(f"End of packet, returning zeros at i={self.i}")
                self.i += 1
                self.y = np.zeros(self.Lfft//2-1, dtype=complex)
                return self.start_flag, self.start_index, self.y, self.i, self.Eq
        return self.start_flag, self.start_index, self.y, self.i, self.Eq