import threading
import numpy as np
import pyaudio
import queue
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend
import matplotlib.pyplot as plt

from config import get_config
from vlc_receiver import OFDMReceiver

# -------- Load config & init receiver --------
cfg = get_config()
RATE = 44100
Lfft = cfg['Lfft']
CP = cfg['cp_length']
CHUNK = (Lfft + CP) * cfg['oversampling_factor']  # samples per audio read
receiver = OFDMReceiver(**cfg)

# -------- State for visualization and stats --------
latest_symbols = np.array([], dtype=complex)
recovered_bits = []
ber_text_value = "BER: --"
errs_text_value = "Errors: --"
dropped_frames = 0
running = True

# Add equalizer state (thread-safe) - using same approach as read_wav.py
import threading
eq_lock = threading.Lock()
# Use N_IDX = 17 like in read_wav.py
N_IDX = 17
latest_eqs = [None] * N_IDX       # Eq per i
latest_x1_by_i = [None] * N_IDX   # store last x1 (previous chunk) per i
latest_x2_by_i = [None] * N_IDX   # store last x2 (current chunk) per i
latest_start_by_i = [None] * N_IDX# store detected start_index relative to x2 per i
latest_y_by_i = [None] * N_IDX    # store last y per i
# Global list of detected start indices (use this for vertical lines)
indexes = []                      # will be appended with start_index values (int)
latest_eq = None                 # keep for backward compatibility (optional)
eq_ready = False                 # set True when final equalizer (i==9) has been captured
eq_plotted = False               # set True after we've plotted the equalizer once

# Collect symbols for constellation (like read_wav.py)
expected_sym_len = Lfft // 2 - 1
collected_ys = []                # list to hold y arrays
symbols_needed = 20              # Same as read_wav.py
symbols_collected = False

global vpos
FORMAT = pyaudio.paInt16
CHANNELS = 1
DEVICE_INDEX = 0  # use default input; set to specific index if needed

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=DEVICE_INDEX,
                frames_per_buffer=CHUNK)

audio_queue = queue.Queue(maxsize=100)

def audio_thread():
    global dropped_frames
    zero_buf = np.zeros(CHUNK, dtype=np.float32)
    while running:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            buf = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            # Light centering; TX added DC offset, so remove mean
            buf -= np.mean(buf)
            audio_queue.put_nowait(buf)
        except queue.Full:
            dropped_frames += 1
        except Exception as e:
            # keep pipeline alive on audio errors by inserting a filler buffer
            print(f"Audio thread read error: {e}; inserting zero buffer")
            try:
                audio_queue.put_nowait(zero_buf)
            except queue.Full:
                dropped_frames += 1
            import time
            time.sleep(0.01)

def receiver_thread():
    global vpos
    global latest_symbols
    global latest_eqs, latest_y_by_i, latest_x1_by_i, latest_x2_by_i, latest_start_by_i, latest_eq, eq_ready
    global collected_ys, symbols_collected, symbols_needed, indexes
    x1 = np.zeros(CHUNK, dtype=np.float32)
    MAX_SYM_WINDOW = 5000
    chunk_idx = 0
    
    while running:
        try:
            x2 = audio_queue.get(timeout=0.2)
        except queue.Empty:
            # no input available -> keep visualizer running
            continue

        # process uses current x1 (previous chunk) and current x2
        start_flag, start_index, y, i, Eq = receiver.process(x1, x2)

        # Capture latest x1, x2, start_index and y and Eq for this i (using read_wav.py approach)
        if 0 <= i < N_IDX:
            with eq_lock:
                # Only store x1, x2 when we have valid signal processing
                should_store_signal = (start_flag or 
                                     (isinstance(y, np.ndarray) and y.size > 0) or
                                     (Eq is not None and getattr(Eq, "size", 0) > 0))
                
                if should_store_signal:
                    try:
                        latest_x1_by_i[i] = x1.copy()
                        latest_x2_by_i[i] = x2.copy()
                    except Exception:
                        pass

                # record most recent start_index values into global indexes list when flagged
                if start_flag and isinstance(start_index, (int, np.integer)) and int(start_index) >= 0:
                    # append under lock and keep list bounded to avoid unbounded growth
                    indexes.append(int(start_index))
                    if len(indexes) > 1000:
                        indexes.pop(0)
                    # store per-i for this specific iteration
                    latest_start_by_i[i] = int(start_index)
                    # Only print for iterations 0 through 15
                    if i <= 10:
                        print(f"Chunk {chunk_idx}, Iteration i={i}: Detected start_index = {int(start_index)}")
                elif start_flag:
                    # Still update even if start_index is not valid, for completeness
                    latest_start_by_i[i] = start_index
                    # Only print for iterations 0 through 15
                    if i <= 15:
                        print(f"Chunk {chunk_idx}, Iteration i={i}: start_flag=True but start_index={start_index} (invalid)")

                if isinstance(y, np.ndarray) and y.size > 0:
                    latest_y_by_i[i] = y.copy()
                    
                if Eq is not None and getattr(Eq, "size", 0) > 0:
                    latest_eqs[i] = Eq.copy()
                    # Use i==9 like read_wav.py for final equalizer
                    if i == 9:
                        latest_eq = Eq.copy()
                        eq_ready = True
                        print("Final channel equalizer Eq computed.")
                        try:
                            vpos = receiver.start_index
                        except Exception:
                            vpos = None
                            
        # If final equalizer has been seen, collect up to symbols_needed y-vectors (each of expected_sym_len)
        # Use same logic as read_wav.py but exclude last iteration
        if (eq_ready and (not symbols_collected) and isinstance(y, np.ndarray) and 
            y.size == expected_sym_len and i < N_IDX-1):
            with eq_lock:
                if len(collected_ys) < symbols_needed:
                    collected_ys.append(y.copy())
                    if len(collected_ys) >= symbols_needed:
                        symbols_collected = True
                        print(f"Collected {symbols_needed} symbols at chunk {chunk_idx}")

        # then advance x1 for next iteration
        x1 = x2
        chunk_idx += 1

        if isinstance(y, np.ndarray) and y.size > 0:
            # accumulate recent symbols for plotting (bounded window)
            if latest_symbols.size == 0:
                latest_symbols = y.copy()
            else:
                latest_symbols = np.concatenate((latest_symbols, y))
                if latest_symbols.size > MAX_SYM_WINDOW:
                    latest_symbols = latest_symbols[-MAX_SYM_WINDOW:]

# Start threads
t_audio = threading.Thread(target=audio_thread, daemon=True)
t_recv = threading.Thread(target=receiver_thread, daemon=True)
t_audio.start()
t_recv.start()

# Show "Waiting for Packet" until we've collected the required symbols
fig_wait = plt.figure("Status", figsize=(4, 2))
ax_wait = fig_wait.add_subplot(111)
ax_wait.axis('off')
status_text = ax_wait.text(0.5, 0.5, "Waiting for Packet...", ha='center', va='center', fontsize=14)

try:
    # block in a GUI-friendly way until symbols_collected becomes True
    while not symbols_collected:
        plt.pause(0.1)

    # collected -> close waiting window
    plt.close(fig_wait)
    print(f"\nProcessing complete. Collected {len(collected_ys)} symbols.")

    if len(collected_ys) == 0:
        print("No symbols collected. Exiting.")
        exit(0)

    # copy collected y-vectors and other state under lock (using read_wav.py approach)
    with eq_lock:
        eqs_copy = [e.copy() if e is not None else None for e in latest_eqs]
        # build concatenated x1||x2 per i
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
        # copy the global indexes for plotting (may be empty)
        indexes_copy = list(indexes)
        ys_last = [y.copy() if isinstance(y, np.ndarray) else None for y in latest_y_by_i]

    # Plot consolidated constellation of the collected y-vectors (using read_wav.py style)
    all_y = np.concatenate(collected_ys) if len(collected_ys) > 0 else np.array([], dtype=complex)

    fig_cons = plt.figure("Collected Constellation", figsize=(8, 8))
    ax_cons = fig_cons.add_subplot(111)
    if all_y.size > 0:
        # Plot constellation points with raw values (no normalization - like read_wav.py)
        real_vals = np.real(all_y)
        imag_vals = np.imag(all_y)
        
        # Print debug info about the raw constellation values
        print(f"\nRaw constellation statistics:")
        print(f"  Real: min={np.min(real_vals):.2f}, max={np.max(real_vals):.2f}, mean={np.mean(real_vals):.2f}")
        print(f"  Imag: min={np.min(imag_vals):.2f}, max={np.max(imag_vals):.2f}, mean={np.mean(imag_vals):.2f}")
        
        ax_cons.scatter(real_vals, imag_vals, s=40, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Set axis limits based on actual raw data range with some padding
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

    # Figure 1: Equalizers (magnitude dB + phase) stacked per i (same as read_wav.py)
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
            # combine legends
            lines, labels = ax_eq.get_legend_handles_labels()
            lines2, labels2 = ax_phase.get_legend_handles_labels()
            if lines or lines2:
                ax_eq.legend(lines + lines2, labels + labels2, fontsize='small', loc='best')
        else:
            ax_eq.text(0.5, 0.5, "no EQ", ha='center', va='center', transform=ax_eq.transAxes)
        ax_eq.set_title(f"i={idx} EQ")

    # Figure 2: Signals (x1||x2 and y real/imag) stacked per i (same style as read_wav.py)
    fig_sig, axes_sig = plt.subplots(N_IDX, 2, figsize=(12, max(6, 1.2 * N_IDX)), num=f"Signals i=0..{N_IDX-1}")
    for idx in range(N_IDX):
        ax_x1 = axes_sig[idx, 0] if N_IDX > 1 else axes_sig[0]
        ax_y = axes_sig[idx, 1] if N_IDX > 1 else axes_sig[1]

        xvals = xs_concat[idx]
        if xvals is not None and getattr(xvals, "size", 0) > 0:
            t = np.arange(len(xvals))
            ax_x1.plot(t, xvals, '-', lw=1)
            ax_x1.set_xlim(0, len(xvals) - 1)
            
            # Plot vertical line for start index detected for this specific iteration (idx)
            x1_len = len(latest_x1_by_i[idx]) if isinstance(latest_x1_by_i[idx], np.ndarray) else 0
            
            # First, plot the specific start index for this iteration if available
            start_idx_for_this_i = latest_start_by_i[idx]
            if start_idx_for_this_i is not None and isinstance(start_idx_for_this_i, (int, np.integer)):
                pos = int(start_idx_for_this_i)
                if 0 <= pos < len(xvals):
                    ax_x1.axvline(pos, color='red', linestyle='-', linewidth=2.5, alpha=0.8, 
                                 label=f'Start idx={start_idx_for_this_i}', zorder=20)
            
            # Also draw recent start indices from all iterations for context (lighter)
            MAX_LINES = 10
            for line_num, s in enumerate(indexes_copy[-MAX_LINES:]):  # Last 10 detections
                if not isinstance(s, (int, np.integer)):
                    continue
                pos = x1_len + int(s)
                if pos < 0 or pos >= len(xvals):
                    continue
                # Make older lines more transparent
                alpha_val = 0.1 + 0.3 * (line_num / max(1, MAX_LINES - 1))
                ax_x1.axvline(pos, color='orange', linestyle='--', linewidth=1.0, 
                             alpha=alpha_val, zorder=10)
            
            # Add legend if there's a main start line
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
            # Include labels like read_wav.py (different from main.py)
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
    plt.show(block=True)  # Block until windows are closed like read_wav.py

finally:
     # Cleanup: stop threads and audio
     running = False
     try:
         stream.stop_stream()
         stream.close()
     except Exception as e:
         print(f"Stream cleanup error: {e}")
     try:
         p.terminate()
     except Exception as e:
         print(f"Error terminating PyAudio: {e}")
         print(f"Stopped. Dropped frames: {dropped_frames}")

print("\nProcessing complete. Plot windows closed.")