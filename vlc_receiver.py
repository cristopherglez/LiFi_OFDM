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
        self.sto_correction = 0.0
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
        correlation_values = correlate(signal_norm, sts_norm, mode='valid')
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
        #received_symbol_no_cp = np.real(received_symbol_no_cp)/np.max(np.abs(received_symbol_no_cp))
        # Print lengths
        #print(f"Length of LTS (no CP): {len(self.lts_no_cp)}")
        #print(f"Length of received symbol (no CP): {len(received_symbol_no_cp)}")
        #delta = np.zeros(self.Lfft * self.oversampling_factor)
        #delta[1] = 1
        #print(f"Delta length: {len(delta)}")
        #data = np.ones_like(self.Nsub, dtype=complex)*(1 + 1j)
        data = np.zeros(self.Nsub+1, dtype=complex)
        """data[:self.Nsub//4] = 1 + 1j
        data[self.Nsub//4:self.Nsub//2] = -1 + 1j
        data[self.Nsub//2:3*self.Nsub//4] = -1 -1j    
        data[3*self.Nsub//4:] = 1 - 1j
        data = np.concatenate((np.zeros(1, dtype=complex), data))
        data_ask = np.conj(np.flip(data))
        new_data = np.concatenate((data, data_ask))
        print(f"Data length: {len(new_data)}")"""
        data[1:] = 4 + 0j 
        #X = np.fft.fft(np.real(self.lts_no_cp), n=self.Lfft)[1:self.Nsub+1]
        #X = np.fft.fft(delta, n=self.Lfft)[1:self.Lfft // 2]
        spectrum = np.zeros(self.Nsub + 1, dtype=complex)
        spectrum[1:] = self.recover_dco_ofdm(received_symbol_no_cp)
        Eq = data[1:] / spectrum[1:]
        #print(f"Eq length: {len(Eq)}")
        return Eq

    def recover_dco_ofdm(self, input_symbol_no_cp):
        decimated_input = []
        for i in range(0, self.Lfft):
            new_index = int(i*self.oversampling_factor*(1+self.normalized_sfo))
            decimated_input.append(input_symbol_no_cp[new_index])
        print(f"Decimated input length: {len(decimated_input)}")
        data = np.fft.fft(decimated_input)[1:self.Nsub+1]
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
        return sto_index, np.sum(minn_metric), minn_metric

    def rzc_method_sto_estimation(self, received_signal):
        corr_length = self.window_length + (self.cp_length * self.oversampling_factor) - 1
        metric = np.zeros(corr_length, dtype=complex)
        Q = np.zeros(corr_length, dtype=complex)
        P = np.zeros(corr_length, dtype=complex)
        R = np.zeros(corr_length, dtype=complex)
        L = self.Lfft*self.oversampling_factor//4
        for d in range(corr_length - 1):
            a_1 = received_signal[d:d + L]
            a_2 = received_signal[d + L: d + 2*L]
            a_3 = received_signal[d + 2*L: d + 3*L]
            a_4 = received_signal[d + 3*L: d + 4*L]
            p = np.sum(np.vdot(a_1, a_2) + np.vdot(a_3, a_4))
            q = np.sum(np.vdot(a_1, a_4))
            r = np.sum(np.vdot(a_2, a_3))
            if len(a_1) < L or len(a_2) < L:
                print(f"Insufficient length for calculation at index {d}")
                metric[d] = 0
                continue
            P[d] = p
            Q[d] = q
            R[d] = r
            metric[d] = np.abs(r)**2*(q**2+p)
        sto_index = int(np.argmax(np.abs(metric)))
        maximum = metric[sto_index]
        return sto_index, maximum, metric

    def frequency_domain_SFO_estimation(self, received_signal, n: int = 0):
        # Compute FFT
        print(f"Length of received signal for SFO estimation: {len(received_signal)}")
        spectrum = np.fft.fft(received_signal, n = self.Lfft * self.oversampling_factor)
        magnitudes = np.abs(spectrum)
        peak_bin = np.argmax(magnitudes[:self.Lfft*self.oversampling_factor//2])
        if self.oversampling_factor == 1:
            f=peak_bin/(self.Lfft*self.oversampling_factor)
            #print(f"Relative frequency: {f} (normalized to Nyquist max = pi)")
            if n != 0:
                print(f"Peak frequency detected: {peak_bin}, real frequency: {n}")
                deviation = peak_bin - n
            else:
                print(f"Peak frequency detected: {peak_bin}, real frequency: {(self.Lfft//2)-1}")
                deviation = peak_bin - ((self.Lfft//2)-1)
        else:
            f=peak_bin/(self.Lfft*self.oversampling_factor)
            #print(f"Relative frequency: {f} (normalized to Nyquist max = pi)")
            if n != 0:
                print(f"Peak frequency detected: {peak_bin}, real frequency: {n}")
                deviation = n - peak_bin
            else:
                print(f"Peak frequency detected: {peak_bin}, real frequency: {(self.Lfft//2)-1}")
                deviation = peak_bin - ((self.Lfft//2)-1)
        T_t = self.Lfft*self.oversampling_factor/n
        T_r = self.Lfft*self.oversampling_factor/peak_bin
        #print(f"Transmitter symbol period: {T_t} samples, Receiver symbol period: {T_r} samples")
        estimated_samples = (T_r/T_t) * self.Lfft * self.oversampling_factor
        #print(f"Estimated samples per symbol at receiver: {estimated_samples} samples")
        #print(f"Frequency deviation: {deviation} bins")
        sample_deviation = estimated_samples - self.Lfft*self.oversampling_factor
        print(f"Sample deviation: {sample_deviation} samples over {(self.Lfft)*self.oversampling_factor} samples")
        buffer_sample_deviation = sample_deviation*(1+(self.cp_length/self.Lfft))
        print(f"Sample deviation: {buffer_sample_deviation} samples over {(self.Lfft+self.cp_length)*self.oversampling_factor} samples")
        return buffer_sample_deviation, magnitudes, spectrum, peak_bin

    def index_correction(self):
        self.sto_acc += self.sto_frac
        if self.sto_acc >= 1.0:
            self.sto_frac_corr = 1
            self.sto_acc -= 1.0
        else:
            self.sto_frac_corr = 0
        if self.start_index + self.sto_int + self.sto_frac_corr < 0:
            self.start_index += int(self.sto_int) + self.sto_frac_corr + self.window_length
            print("ERROR: Start index negative after SFO correction, skipping symbol")
            print(f"New start index: {self.start_index}")
        else: 
            self.start_index += int(self.sto_int) + int(self.sto_frac_corr)
        pass

    def interpolate_correction(self, signal):
        length_with_sfo = (self.Lfft * self.oversampling_factor)
        real_length = length_with_sfo - self.sto_correction
        chunk = signal[self.start_index : self.start_index + int(length_with_sfo)]
        chunk = np.interp(np.linspace(0, len(chunk), int(real_length), endpoint=False),
                              np.arange(len(chunk)), chunk)
        return chunk

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
                sfo_deviation_try, _, _, _= self.frequency_domain_SFO_estimation(x1, int(self.Lfft*self.oversampling_factor//2.5))
                self.sto_correction += sfo_deviation_try
                self.y = np.fft.fft(x1)
                self.i += 1
                return self.start_flag, self.start_index, self.y, self.i, self.Eq
            elif self.i == (self.sfo_repetitions):
                # Perform SFO calculation
                self.sto_correction /= self.sfo_repetitions
                self.sto_correction = -2.1
                print(f"Samples offset per buffer: {self.sto_correction}")
                self.sto_int = int(self.sto_correction)
                print(f"Integer STO correction (samples): {self.sto_int}")
                self.sto_frac = self.sto_correction - self.sto_int
                print(f"Fractional STO correction (samples): {self.sto_frac}")
                self.sto_frac_corr = 0
                print(f"STO correction (samples): {self.sto_correction}")
                self.index_correction()
                self.i += 1
                return self.start_flag, self.start_index, self.y, self.i, self.Eq
            elif (self.i >= (self.sfo_repetitions + 1)) and (self.i < (self.sfo_repetitions + 2)):
                ######
                new_index , new_minn_value, minn_metric = self.minn_method_sto_estimation(signal)
                print(f"Minn's STO estimation for: {self.i}, metric value: {new_minn_value}")
                if np.abs(new_minn_value) > np.abs(self.minn_value):
                    print(f"Minn's metric improved from {self.minn_value} to {new_minn_value}")
                    print(f"Updated Minn's STO estimation from {self.start_index} to {new_index}")
                    self.start_index = new_index
                    self.minn_value = new_minn_value
                else:
                    self.index_correction()
                    self.i += 1
                if self.start_index > self.window_length:
                    self.start_index -= self.window_length
                print(f"Start index after Minn's STO estimation: {self.start_index}")
                print(f"i={self.i}")
                self.y = minn_metric
                ######
                """self.index_correction()
                if self.i == (self.sfo_repetitions + 1):
                    self.corrected_length = int(len(signal) * (1 - self.normalized_sfo))
                    print(f"real length of signal for Minn's STO estimation: {len(signal)}")
                    print(f"corrected length of signal for Minn's STO estimation: {self.corrected_length}")
                    print(f"difference in length: {len(signal) - self.corrected_length}")
                    print(f"Calculated sto_correction: {self.sto_correction}, sto_int: {self.sto_int}, sto_frac: {self.sto_frac}")
                new_index , new_minn_value, minn_metric = self.minn_method_sto_estimation(signal)
                print(f"Minn's STO estimation for: {self.i}, metric value: {new_minn_value}")
                if np.abs(new_minn_value) > np.abs(self.minn_value):
                    print(f"Minn's metric improved from {self.minn_value} to {new_minn_value}")
                    print(f"Updated Minn's STO estimation from {self.start_index} to {new_index}")
                    self.start_index = new_index
                    self.minn_value = new_minn_value
                else:
                    self.i += 1
                if self.start_index > self.window_length:
                    self.start_index -= self.window_length
                print(f"Start index after Minn's STO estimation: {self.start_index}")
                print(f"i={self.i}")"""
                return self.start_flag, self.start_index, self.y, self.i, self.Eq
            elif(self.i >= self.sfo_repetitions + 2) and (self.i <= self.sfo_repetitions + self.lts_repetitions):
                self.index_correction()
                chunk = self.interpolate_correction(signal)
                #chunk = signal[self.start_index : self.start_index + self.Lfft * self.oversampling_factor]
                if self.i == self.sfo_repetitions + 1:
                    self.i += 1
                    return self.start_flag, self.start_index, self.y, self.i, self.Eq
                else:
                    # Perform channel estimation
                    self.Eq += self.channel_estimation_ls(chunk)
                    self.y = chunk
                    self.i += 1
                    return self.start_flag, self.start_index, self.y, self.i, self.Eq
            elif self.i == self.lts_repetitions + self.sfo_repetitions + 1:
                # SFO corrections
                self.index_correction()
                chunk = self.interpolate_correction(signal)
                #chunk = signal[self.start_index : self.start_index + self.Lfft * self.oversampling_factor]
                self.Eq =+ self.channel_estimation_ls(chunk)
                # Finalize LTS estimation
                self.Eq = self.Eq / (self.lts_repetitions)
                #self.Eq = np.ones(self.Nsub, dtype=complex)  # TEMPORARY SET TO ONES FOR TESTING
                print(f"Final channel equalizer Eq computed.")
                self.y = chunk
                self.i += 1
                return self.start_flag, self.start_index, self.y, self.i, self.Eq
            elif self.i <= self.lts_repetitions + self.sfo_repetitions + self.data_frame_length + 1:
                # SFO corrections
                self.index_correction()
                # Correct interpolating
                if self.start_index + self.Lfft * self.oversampling_factor > len(signal)*(1 - self.normalized_sfo):
                    print("Not enough samples for data frame processing.")
                    self.i += 1
                    return self.start_flag, self.start_index, self.y, self.i, self.Eq
                else:
                    chunk = self.interpolate_correction(signal)
                    chunk = signal[self.start_index : self.start_index + self.Lfft * self.oversampling_factor]
                    # Process data frames
                    self.y = self.recover_dco_ofdm(chunk) * self.Eq
                    self.i += 1
                    return self.start_flag, self.start_index, self.y, self.i, self.Eq
        # Ensure a value is always returned (prevent caller unpacking None)
        return self.start_flag, self.start_index, self.y, self.i, self.Eq

    def process_no_sfo(self, x1, x2):
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
                sfo_deviation_try, _, _, _ = self.frequency_domain_SFO_estimation(x1, self.Nsub)
                self.sfo_deviation += sfo_deviation_try
                #self.y = np.fft.fft(x1, n=self.Lfft)
                self.i += 1
                return self.start_flag, self.start_index, self.y, self.i, self.Eq
            elif self.i == (self.sfo_repetitions):
                # Set SFO to zero
                self.sfo_deviation = 0.0
                self.normalized_sfo = 0
                self.sto_correction = 0.0
                self.sto_int = int(self.sto_correction)
                self.sto_frac = self.sto_correction - self.sto_int
                self.sto_frac_corr = 0
                print(f"STO correction (samples): {self.sto_correction}")
                self.i += 1
                return self.start_flag, self.start_index, {}, self.i, self.Eq
            elif (self.i >= (self.sfo_repetitions + 1)) and (self.i < (self.sfo_repetitions + 2)):
                new_index , new_minn_value, minn_metric = self.rzc_method_sto_estimation(signal)
                print(f"Minn's STO estimation for: {self.i}, metric value: {new_minn_value}")
                if np.abs(new_minn_value) > np.abs(self.minn_value):
                    print(f"Minn's metric improved from {self.minn_value} to {new_minn_value}")
                    print(f"Updated Minn's STO estimation from {self.start_index} to {new_index}")
                    self.start_index = new_index
                    self.minn_value = new_minn_value
                else:
                    self.i += 1
                if self.start_index > self.window_length:
                    self.start_index -= self.window_length
                print(f"Start index after Minn's STO estimation: {self.start_index}")
                print(f"i={self.i}")
                #self.y = minn_metric
                return self.start_flag, self.start_index, self.y, self.i, self.Eq
            elif(self.i >= self.sfo_repetitions + 2) and (self.i <= self.sfo_repetitions + self.lts_repetitions):
                #self.index_correction()
                #chunk = self.interpolate_correction(signal)
                chunk = signal[self.start_index : self.start_index + self.Lfft * self.oversampling_factor]
                if self.i == self.sfo_repetitions + 1:
                    self.i += 1
                    return self.start_flag, self.start_index, self.y, self.i, self.Eq
                else:
                    # Perform channel estimation
                    self.Eq += self.channel_estimation_ls(chunk)
                    self.y = chunk
                    self.i += 1
                    return self.start_flag, self.start_index, self.y, self.i, self.Eq
            elif self.i == self.lts_repetitions + self.sfo_repetitions + 1:
                # SFO corrections
                #self.index_correction()
                #chunk = self.interpolate_correction(signal)
                chunk = signal[self.start_index : self.start_index + self.Lfft * self.oversampling_factor]
                self.Eq = self.channel_estimation_ls(chunk)
                # Finalize LTS estimation
                self.Eq = self.Eq / (self.lts_repetitions)                
                print(f"Final channel equalizer Eq computed.")
                self.y = chunk
                self.i += 1
                return self.start_flag, self.start_index, self.y, self.i, self.Eq
            elif self.i <= self.lts_repetitions + self.sfo_repetitions + self.data_frame_length + 2:
                # SFO corrections
                self.index_correction()
                # Correct interpolating
                if self.start_index + self.Lfft * self.oversampling_factor > len(signal)*(1 - self.normalized_sfo):
                    print("Not enough samples for data frame processing.")
                    self.i += 1
                    return self.start_flag, self.start_index, self.y, self.i, self.Eq
                else:
                    #chunk = self.interpolate_correction(signal)
                    chunk = signal[self.start_index : self.start_index + self.Lfft * self.oversampling_factor]
                    # Process data frames
                    print(f"Eq length: {len(self.Eq)}")
                    data = self.recover_dco_ofdm(chunk)
                    print(f"Data length: {len(data)}")
                    self.y = self.recover_dco_ofdm(chunk) * self.Eq
                    self.i += 1
                    return self.start_flag, self.start_index, self.y, self.i, self.Eq
        # Ensure a value is always returned (prevent caller unpacking None)
        return self.start_flag, self.start_index, self.y, self.i, self.Eq