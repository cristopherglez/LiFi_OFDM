import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.signal

from read_wav_config import get_wav_config
from vlc_receiver import OFDMReceiver

# -------- Load config & init receiver --------
cfg = get_wav_config()
RATE = cfg['sample_rate']
Lfft = cfg['Lfft']
CP = cfg['cp_length']
CHUNK = (Lfft + CP) * cfg['oversampling_factor']  # samples per audio read
SFO = -0.035  # Example SFO value for receiver
SFO_ON = False  # Enable/Disable SFO correction for this test
input_signal = cfg['input_signal']
SNR = 600.0
if SNR < 1000: # Not added above this level of SNR_dB
    signal_power = np.mean(input_signal**2)
    noise_power = signal_power / (10**(SNR / 10))
    noise = np.sqrt(noise_power) * np.random.randn(len(input_signal))
    input_signal = input_signal + noise
if SFO_ON == True:
    new_length = int(len(input_signal) * (1 + SFO))
    x_original = np.arange(len(input_signal))
    x_new = np.linspace(0, len(input_signal) - 1, new_length)
    input_signal = np.interp(x_new, x_original, input_signal)


# Remove non-receiver params from config
receiver_cfg = {k: v for k, v in cfg.items() if k not in ['input_signal', 'sample_rate']}
receiver = OFDMReceiver(**receiver_cfg)

# Load input signal from WAV file
print(f"Processing {len(input_signal)} samples from WAV file...")

# -------- State for visualization --------
N_IDX = 17
expected_sym_len = Lfft // 2 - 1
symbols_needed = 20

# Storage for plotting
latest_eqs = [None] * N_IDX
latest_x1_by_i = [None] * N_IDX
latest_x2_by_i = [None] * N_IDX
latest_start_by_i = [None] * N_IDX
latest_y_by_i = [None] * N_IDX
indexes = []
collected_ys = []
eq_ready = False

# -------- Process WAV file in chunks --------
x1 = np.zeros(CHUNK, dtype=np.float32)

# Calculate total chunks to process
total_samples = len(input_signal)
num_chunks = total_samples // CHUNK
print(f"Total chunks to process: {num_chunks}")

for chunk_idx in range(num_chunks):
    start_idx = chunk_idx * CHUNK
    end_idx = start_idx + CHUNK
    
    if end_idx > total_samples:
        # Pad last chunk if needed
        x2 = np.zeros(CHUNK, dtype=np.float32)
        remaining = total_samples - start_idx
        x2[:remaining] = input_signal[start_idx:total_samples]
    else:
        x2 = input_signal[start_idx:end_idx]
    
    # Process chunk
    if SFO_ON:
        start_flag, start_index, y, i, Eq = receiver.process(x1, x2)
    else:
        start_flag, start_index, y, i, Eq = receiver.process_no_sfo(x1, x2)
    
    # Capture state per iteration i
    if 0 <= i < N_IDX:
        should_store_signal = (start_flag or 
                             (isinstance(y, np.ndarray) and y.size > 0) or
                             (Eq is not None and getattr(Eq, "size", 0) > 0))
        
        if should_store_signal:
            try:
                latest_x1_by_i[i] = x1.copy()
                latest_x2_by_i[i] = x2.copy()
            except Exception:
                pass
        
        if start_flag and isinstance(start_index, (int, np.integer)) and int(start_index) >= 0:
            indexes.append(int(start_index))
            if len(indexes) > 1000:
                indexes.pop(0)
            latest_start_by_i[i] = int(start_index)
            if i <= 15:
                print(f"Chunk {chunk_idx}, Iteration i={i}: Detected start_index = {int(start_index)}")
        
        if isinstance(y, np.ndarray) and y.size > 0:
            latest_y_by_i[i] = y.copy()
        
        if Eq is not None and getattr(Eq, "size", 0) > 0:
            latest_eqs[i] = Eq.copy()
            if i == 9:
                eq_ready = True
                print("Final channel equalizer Eq computed.")
    
    # Collect y-vectors for constellation (after equalizer is ready)
    if eq_ready and len(collected_ys) < symbols_needed and isinstance(y, np.ndarray) and y.size == expected_sym_len:
        collected_ys.append(y.copy())
        if len(collected_ys) >= symbols_needed:
            print(f"Collected {symbols_needed} symbols at chunk {chunk_idx}")
    
    # Advance x1 for next iteration
    x1 = x2

print(f"\nProcessing complete. Collected {len(collected_ys)} symbols.")

if len(collected_ys) == 0:
    print("No symbols collected. Exiting.")
    exit(0)

# Copy collected data
eqs_copy = [e.copy() if e is not None else None for e in latest_eqs]
xs_concat = []
for idx in range(N_IDX):
    x1c = latest_x1_by_i[idx]
    x2c = latest_x2_by_i[idx]
    if isinstance(x1c, np.ndarray) and isinstance(x2c, np.ndarray):
        concat = np.concatenate((x1c, x2c))
    elif isinstance(x1c, np.ndarray):
        concat = x1c.copy()
    elif isinstance(x2c, np.ndarray):
        concat = x2c.copy()
    else:
        concat = None
    xs_concat.append(concat)
indexes_copy = list(indexes)
ys_last = [y.copy() if isinstance(y, np.ndarray) else None for y in latest_y_by_i]

# Plot consolidated constellation
all_y = np.concatenate(collected_ys) if len(collected_ys) > 0 else np.array([], dtype=complex)

fig_cons = plt.figure("Collected Constellation", figsize=(8, 8))
ax_cons = fig_cons.add_subplot(111)
if all_y.size > 0:
    real_vals = np.real(all_y)
    imag_vals = np.imag(all_y)
    
    print(f"\nRaw constellation statistics:")
    print(f"  Real: min={np.min(real_vals):.2f}, max={np.max(real_vals):.2f}, mean={np.mean(real_vals):.2f}")
    print(f"  Imag: min={np.min(imag_vals):.2f}, max={np.max(imag_vals):.2f}, mean={np.mean(imag_vals):.2f}")
    
    ax_cons.scatter(real_vals, imag_vals, s=40, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    real_range = np.max(real_vals) - np.min(real_vals)
    imag_range = np.max(imag_vals) - np.min(imag_vals)
    padding = max(real_range, imag_range) * 0.1
    
    ax_cons.set_xlim(np.min(real_vals) - padding, np.max(real_vals) + padding)
    ax_cons.set_ylim(np.min(imag_vals) - padding, np.max(imag_vals) + padding)
else:
    ax_cons.text(0.5, 0.5, "no data", ha='center', va='center', transform=ax_cons.transAxes)
    ax_cons.set_xlim(-2, 2)
    ax_cons.set_ylim(-2, 2)

ax_cons.set_title(f"Collected {len(collected_ys)} y-vectors ({len(all_y)} symbols)")
ax_cons.set_xlabel("In-phase")
ax_cons.set_ylabel("Quadrature")
ax_cons.grid(True, alpha=0.3)
ax_cons.set_aspect('equal')

# Figure 1: Equalizers (magnitude dB + phase) stacked per i
fig_eq, axes_eq = plt.subplots(N_IDX, 1, figsize=(10, max(6, 0.6 * N_IDX)), num=f"Equalizers i=0..{N_IDX-1}")
for idx in range(N_IDX):
    ax_eq = axes_eq[idx] if N_IDX > 1 else axes_eq
    eq = eqs_copy[idx]
    if eq is not None:
        idxs = np.arange(len(eq))
        mag_db = 20.0 * np.log10(np.abs(eq) + 1e-12)
        phase = np.angle(eq)
        ax_eq.plot(idxs, mag_db, '-o', markersize=3, color='C0', label='mag (dB)')
        ax_eq.set_ylabel("Mag (dB)", color='C0')
        ax_eq.tick_params(axis='y', colors='C0')
        ax_phase = ax_eq.twinx()
        ax_phase.plot(idxs, phase, '-', color='C1', label='phase (rad)')
        ax_phase.set_ylabel("Phase (rad)", color='C1')
        ax_phase.tick_params(axis='y', colors='C1')
        lines, labels = ax_eq.get_legend_handles_labels()
        lines2, labels2 = ax_phase.get_legend_handles_labels()
        if lines or lines2:
            ax_eq.legend(lines + lines2, labels + labels2, fontsize='small', loc='best')
    else:
        ax_eq.text(0.5, 0.5, "no EQ", ha='center', va='center', transform=ax_eq.transAxes)
    ax_eq.set_title(f"i={idx} EQ")

# Figure 2: Signals (x1||x2 and y real/imag) stacked per i
fig_sig, axes_sig = plt.subplots(N_IDX, 2, figsize=(12, max(6, 1.2 * N_IDX)), num=f"Signals i=0..{N_IDX-1}")
for idx in range(N_IDX):
    ax_x1 = axes_sig[idx, 0] if N_IDX > 1 else axes_sig[0]
    ax_y = axes_sig[idx, 1] if N_IDX > 1 else axes_sig[1]
    
    xvals = xs_concat[idx]
    if xvals is not None and getattr(xvals, "size", 0) > 0:
        t = np.arange(len(xvals))
        ax_x1.plot(t, xvals, '-', lw=1)
        ax_x1.set_xlim(0, len(xvals) - 1)
        
        x1_len = len(latest_x1_by_i[idx]) if isinstance(latest_x1_by_i[idx], np.ndarray) else 0
        
        start_idx_for_this_i = latest_start_by_i[idx]
        if start_idx_for_this_i is not None and isinstance(start_idx_for_this_i, (int, np.integer)):
            pos = int(start_idx_for_this_i)
            if 0 <= pos < len(xvals):
                ax_x1.axvline(pos, color='red', linestyle='-', linewidth=2.5, alpha=0.8, 
                             label=f'Start idx={start_idx_for_this_i}', zorder=20)
        
        MAX_LINES = 10
        for line_num, s in enumerate(indexes_copy[-MAX_LINES:]):
            if not isinstance(s, (int, np.integer)):
                continue
            pos = x1_len + int(s)
            if pos < 0 or pos >= len(xvals):
                continue
            alpha_val = 0.1 + 0.3 * (line_num / max(1, MAX_LINES - 1))
            ax_x1.axvline(pos, color='orange', linestyle='--', linewidth=1.0, 
                         alpha=alpha_val, zorder=10)
        
        if start_idx_for_this_i is not None:
            ax_x1.legend(fontsize='small', loc='upper right')
    else:
        ax_x1.text(0.5, 0.5, "no x1||x2", ha='center', va='center', transform=ax_x1.transAxes)
    ax_x1.set_title(f"i={idx} x1||x2")
    ax_x1.set_xlabel("sample")
    ax_x1.set_ylabel("amp")
    
    yvals = ys_last[idx]
    if yvals is not None and getattr(yvals, "size", 0) > 0:
        t_y = np.arange(len(yvals))
        ax_y.plot(t_y, np.real(yvals), '-', linewidth=1, label='real')
        ax_y.plot(t_y, np.imag(yvals), '-', linewidth=1, label='imag')
        ax_y.set_xlabel("sample")
        ax_y.set_ylabel("value")
        ax_y.legend(fontsize='small')
    else:
        ax_y.text(0.5, 0.5, "no y", ha='center', va='center', transform=ax_y.transAxes)
    ax_y.set_title(f"i={idx} y (real / imag)")

plt.tight_layout()
print("\nAll plots generated. Displaying windows...")
plt.show(block=True)  # Block until windows are closed

print("\nProcessing complete. Plot windows closed.")
