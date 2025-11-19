import numpy as np
from scipy.signal import correlate

class OFDMReceiver:
    def __init__(self, Lfft, cp_length, data_frame_length, lts_repetitions, sfo_repetitions,
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
        self.sfo_repetitions = sfo_repetitions

        # References
        self.sts_no_cp = sts_no_cp
        self.lts_no_cp = lts_no_cp

        # QPSK reference
        self.qpsk_points = np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype=complex)

        # State
        self.start_flag = False
        self.start_index = 0
        self.i = 0
        self.sfo = 0
        self.normalized_sfo = 0
        self.sto=0.0
        self.sto_acc=0.0
        self.Eq = np.zeros(self.Nsub, dtype=complex)
        self.y = np.array([], dtype=complex)
        self.sfo_deviation = 0.0
        self.minn_value = 0.0

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
        correlation_values = correlate(sts_norm, signal_norm, mode='valid')
        peak_value = np.max(np.abs(correlation_values))
        start_index = np.argmax(np.abs(correlation_values)) + self.cp_length*self.oversampling_factor

        
        if start_index < self.window_length:
            start_flag = peak_value > threshold
        else:
            start_flag = False
        # Notify if packet is detected
        if start_flag:
            print(f"Packet detected at index: {start_index}")
        return start_flag, start_index, correlation_values, sts_auto_corr
    

    def channel_estimation_ls(self, received_symbol_no_cp):
        # Print lengths
        #print(f"Length of LTS (no CP): {len(self.lts_no_cp)}")
        #print(f"Length of received symbol (no CP): {len(received_symbol_no_cp)}")
        #delta = np.zeros(self.Lfft, dtype=complex)
        #delta[1] = 1
        #print(f"Delta length: {len(delta)}")
        X = np.fft.fft(np.real(self.lts_no_cp), n=self.Lfft)[1:self.Lfft // 2]
        #X = np.fft.fft(delta, n=self.Lfft)[1:self.Lfft // 2]
        Y = np.fft.fft(received_symbol_no_cp, n=self.Lfft)[1:self.Lfft // 2]
        Eq = X / (Y)  # zero-forcing equalizer, Eq ≈ 1/H
        #print(f"Eq length: {len(Eq)}")
        return Eq

    def recover_dco_ofdm(self, input_symbol_no_cp):
        spectrum = np.fft.fft(input_symbol_no_cp)
        data = spectrum[1:self.Nsub+1]
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
        return sto_index, minn_metric[sto_index],minn_metric,

    def frequency_domain_SFO_estimation(self, received_signal, n: int = 0):
        # Compute FFT
        spectrum = np.fft.fftshift(np.fft.fft(received_signal))
        magnitudes = np.abs(spectrum)[len(spectrum)//2:]
        peak_index = np.argmax(magnitudes)
        if self.oversampling_factor == 1:
            peak_bin = peak_index*(self.Lfft)/((len(received_signal)))
            if n != 0:
                print(f"Peak frequency detected: {peak_bin}, real frequency: {n}")
                deviation = peak_bin - n
            else:
                print(f"Peak frequency detected: {peak_bin}, real frequency: {(self.Lfft//2)-1}")
                deviation = peak_bin - ((self.Lfft//2)-1)
        else:
            peak_bin = peak_index*(self.Lfft* self.oversampling_factor)/((len(received_signal)))
            if n != 0:
                print(f"Peak frequency detected: {peak_bin}, real frequency: {n}")
                deviation = n - peak_bin
            else:
                print(f"Peak frequency detected: {peak_bin}, real frequency: {(self.Lfft//2)-1}")
                deviation = peak_bin - ((self.Lfft//2)-1)

        print(f"Frequency deviation: {deviation} bins")
        sample_deviation = 2*deviation
        print(f"Sample deviation: {sample_deviation} samples over {len(received_signal)} samples")
        return deviation#, magnitudes

    def process(self, x1, x2):
        signal = np.concatenate([x1, x2]) 
        if not self.start_flag:
            # Perform packet detection, coarse sync
            self.start_flag, self.start_index, _, _ = self.packet_detection(signal)
        else: 
            if self.i == 0:
                self.i += 1
                return self.start_flag, self.start_index, self.y, self.i, self.Eq
            # From 1 to number of sfo estimations
            elif self.i > 0 and self.i < (self.sfo_repetitions):
                self.sfo_deviation += self.frequency_domain_SFO_estimation(x1, self.Nsub)
                self.y = np.fft.fft(x1)
                self.i += 1
                return self.start_flag, self.start_index, self.y, self.i, self.Eq
            elif self.i == (self.sfo_repetitions):
                self.i += 1
                return self.start_flag, self.start_index, self.y, self.i, self.Eq
            elif (self.i >= (self.sfo_repetitions + 1)) and (self.i <= (self.sfo_repetitions + 2)):
                if self.i == (self.sfo_repetitions + 1):
                # Perform SFO calculation
                    self.sto_correction = self.sfo_deviation / self.sfo_repetitions
                    self.sto_int = self.sto_correction // 1
                    self.sto_frac = self.sto_correction - self.sto_int
                    self.sto_frac_corr = 0
                    #print(f"STO correction (samples): {self.sto_correction}")
                    self.normalized_sfo = self.sfo_deviation / (self.sfo_repetitions*self.Nsub)
                    print(f"Normalized SFO after final calculation: {self.normalized_sfo}")
                    #print(f"Original length no SFO: {self.Lfft * self.oversampling_factor}")
                    real_length = int(self.Lfft * (1 + self.normalized_sfo) * self.oversampling_factor)
                    #print(f"Actual length with SFO: {real_length}")
                    self.corrected_length = int(len(signal) * (1 - self.normalized_sfo))
                    if self.corrected_length != len(signal):  
                        print(f"Corrected length of signal for Minn's STO estimation: {self.corrected_length}")
                        corrected_signal = np.interp(np.linspace(0, len(signal), self.corrected_length, endpoint=False),
                                                np.arange(len(signal)), signal)
                        # Zero padding or truncating to ensure sufficient length
                        if len(corrected_signal) < self.window_length * 2:
                            pad_length = (self.window_length * 2) - len(corrected_signal)
                            corrected_signal = np.pad(corrected_signal, (0, pad_length), 'constant')
                        elif len(corrected_signal) > self.window_length * 2:
                            corrected_signal = corrected_signal[:self.window_length * 2]
                        signal = corrected_signal
                new_index , new_minn_value, minn_metric = self.minn_method_sto_estimation(signal)
                if new_minn_value > self.minn_value:
                    print(f"Updated Minn's STO estimation from {self.start_index} to {new_index}")
                    self.start_index = new_index
                if self.start_index > self.window_length:
                    self.start_index -= self.window_length
                print(f"Start index after Minn's STO estimation: {self.start_index}")
                self.i += 1
                print(f"i={self.i}")
                return self.start_flag, self.start_index, minn_metric, self.i, self.Eq
            elif (self.i > self.sfo_repetitions + 2) and (self.i <= self.sfo_repetitions + self.lts_repetitions + 2):
                # SFO corrections
                self.sto_acc += self.sto_frac
                if self.sto_acc >= 1.0:
                    self.sto_frac_corr = 1
                    self.sto_acc -= 1.0
                else:
                    self.sto_frac_corr = 0
                if self.start_index + self.sto_int + self.sto_frac_corr < 0:
                    self.start_index += self.sto_int + self.sto_frac_corr + self.window_length
                    print("ERROR: Start index negative after SFO correction, skipping symbol")
                else: 
                    self.start_index += int(self.sto_int) + int(self.sto_frac_corr)
                    # Correct interpolating
                    chunk = signal[self.start_index : self.start_index + self.Lfft * self.oversampling_factor]
                    new_length = self.Lfft
                    chunk = np.interp(np.linspace(0, len(chunk), new_length, endpoint=False),
                                      np.arange(len(chunk)), chunk)
                    # Perform channel estimation
                    self.Eq = self.channel_estimation_ls(chunk)
                self.i += 1
                return self.start_flag, self.start_index, {}, self.i, self.Eq
            elif self.i == self.lts_repetitions + self.sfo_repetitions:
                # SFO corrections
                self.sto_acc += self.sto_frac
                if self.sto_acc >= 1.0:
                    self.sto_frac_corr = 1
                    self.sto_acc -= 1.0
                else:
                    self.sto_frac_corr = 0
                if self.start_index + self.sto_int + self.sto_frac_corr < 0:
                    self.start_index += self.sto_int + self.sto_frac_corr + self.window_length
                    print("ERROR: Start index negative after SFO correction, skipping symbol")
                else: 
                    self.start_index += int(self.sto_int) + int(self.sto_frac_corr)
                # Finalize LTS estimation
                #self.Eq = self.Eq / (self.lts_repetitions)
                self.Eq = np.ones(self.Nsub)
                self.i += 1
                return self.start_flag, self.start_index, {}, self.i, self.Eq
            elif self.i > self.lts_repetitions + self.sfo_repetitions and self.i < self.lts_repetitions + self.sfo_repetitions + self.data_frame_length:
                # SFO corrections
                self.sto_acc += self.sto_frac
                if self.sto_acc >= 1.0:
                    self.sto_frac_corr = 1
                    self.sto_acc -= 1.0
                else:
                    self.sto_frac_corr = 0
                if self.start_index + self.sto_int + self.sto_frac_corr < 0:
                    self.start_index += self.sto_int + self.sto_frac_corr + self.window_length
                    print("ERROR: Start index negative after SFO correction, skipping symbol")
                else: 
                    self.start_index += int(self.sto_int) + int(self.sto_frac_corr)
                    # Correct interpolating
                    chunk = signal[self.start_index : self.start_index + self.Lfft * self.oversampling_factor]
                    new_length = self.Lfft
                    chunk = np.interp(np.linspace(0, len(chunk), new_length, endpoint=False),
                                      np.arange(len(chunk)), chunk)
                    # Perform channel estimation
                    self.Eq += self.channel_estimation_ls(chunk)
                # Correct interpolating
                chunk = signal[self.start_index : self.start_index + self.Lfft * self.oversampling_factor]
                new_length = self.Lfft
                chunk = np.interp(np.linspace(0, len(chunk), new_length, endpoint=False),np.arange(len(chunk)), chunk)
                # Process data frames
                self.y = self.recover_dco_ofdm(signal) * self.Eq
                self.i += 1
                return self.start_flag, self.start_index, self.y, self.i, self.Eq
        # Ensure a value is always returned (prevent caller unpacking None)
        return self.start_flag, self.start_index, self.y, self.i, self.Eq
