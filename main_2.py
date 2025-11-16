import threading
import numpy as np
import pyaudio
import queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

# Add equalizer state (thread-safe)
import threading
eq_lock = threading.Lock()
# keep single latest snapshot per i = 0..19
N_IDX = 20                        # show i = 0..19
latest_eqs = [None] * N_IDX       # Eq per i
latest_x1_by_i = [None] * N_IDX   # store last x1 (previous chunk) per i
latest_x2_by_i = [None] * N_IDX   # store last x2 (current chunk) per i
latest_start_by_i = [None] * N_IDX# store detected start_index relative to x2 per i
latest_y_by_i = [None] * N_IDX    # store last y per i
# New: global list of detected start indices (use this for vertical lines)
indexes = []                      # will be appended with start_index values (int)
# keep for backward compatibility (optional)
latest_eq = None                 # keep for backward compatibility (optional)
eq_ready = False                 # set True when final equalizer (i==5) has been captured
eq_plotted = False               # set True after we've plotted the equalizer once

# NEW: collect exactly 100 y-vectors of expected length (Lfft/2 - 1)
expected_sym_len = Lfft // 2 - 1
collected_ys = []                # list to hold up to 100 y arrays
symbols_needed = 100
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
    while running:
        try:
            x2 = audio_queue.get(timeout=0.2)
        except queue.Empty:
            # no input available -> keep visualizer running
            continue

        # process uses current x1 (previous chunk) and current x2
        start_flag, start_index, y, i, Eq = receiver.process(x1, x2)

        # Capture latest x1, x2, start_index and y and Eq for this i (overwrite latest)
        if 0 <= i < N_IDX:
            with eq_lock:
                try:
                    latest_x1_by_i[i] = x1.copy()
                except Exception:
                    latest_x1_by_i[i] = None
                try:
                    latest_x2_by_i[i] = x2.copy()
                except Exception:
                    latest_x2_by_i[i] = None

                # record most recent start_index values into global indexes list when flagged
                if start_flag and isinstance(start_index, (int, np.integer)) and int(start_index) >= 0:
                    # append under lock and keep list bounded to avoid unbounded growth
                    indexes.append(int(start_index))
                    if len(indexes) > 1000:
                        indexes.pop(0)
                    # also store per-i for reference if needed
                    latest_start_by_i[i] = int(start_index)
                    print(f"Detected start_index for i={i}: {latest_start_by_i[i]}")

                if isinstance(y, np.ndarray) and y.size > 0:
                    latest_y_by_i[i] = y.copy()
                if Eq is not None and getattr(Eq, "size", 0) > 0:
                    latest_eqs[i] = Eq.copy()
                    if i == 5:
                        latest_eq = Eq.copy()
                        eq_ready = True
                        try:
                            vpos = receiver.start_index
                        except Exception:
                            vpos = None
        # If final equalizer has been seen, collect up to symbols_needed y-vectors (each of expected_sym_len)
        if eq_ready and (not symbols_collected) and isinstance(y, np.ndarray) and y.size == expected_sym_len:
            with eq_lock:
                if len(collected_ys) < symbols_needed:
                    collected_ys.append(y.copy())
                    if len(collected_ys) >= symbols_needed:
                        symbols_collected = True

        # then advance x1 for next iteration
        x1 = x2

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

# Show "Waiting for Packet" until we've collected the required 100 y-vectors
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

    # copy collected y-vectors and other state under lock
    with eq_lock:
        ys_copy_list = [y.copy() for y in collected_ys] if len(collected_ys) > 0 else []
        eqs_copy = [e.copy() if e is not None else None for e in latest_eqs]
        # build concatenated x1||x2 per i (no start/end arrays here)
        xs_concat = []
        for idx in range(N_IDX):
            x1c = latest_x1_by_i[idx]
            x2c = latest_x2_by_i[idx]
            if isinstance(x1c, np.ndarray) and isinstance(x2c, np.ndarray):
                concat = np.concatenate((x1c, x2c))
                x1_len = len(x1c)
            elif isinstance(x1c, np.ndarray):
                concat = x1c.copy()
                x1_len = len(x1c)
            elif isinstance(x2c, np.ndarray):
                concat = x2c.copy()
                x1_len = 0
            else:
                concat = None
                x1_len = 0
            xs_concat.append(concat)
        # copy the global indexes for plotting (may be empty)
        indexes_copy = list(indexes)
        ys_last = [y.copy() if isinstance(y, np.ndarray) else None for y in latest_y_by_i]

    # Plot consolidated constellation of the 100 collected y-vectors (flattened)
    if len(ys_copy_list) > 0:
        all_y = np.concatenate(ys_copy_list)
    else:
        all_y = np.array([], dtype=complex)

    fig_cons = plt.figure("Collected Constellation", figsize=(6, 6))
    ax_cons = fig_cons.add_subplot(111)
    if all_y.size > 0:
        ax_cons.plot(np.real(all_y), np.imag(all_y), 'o', markersize=4, alpha=0.6)
    else:
        ax_cons.text(0.5, 0.5, "no data", ha='center', va='center', transform=ax_cons.transAxes)
    ax_cons.set_title(f"Collected {len(ys_copy_list)} y-vectors ({len(all_y)} symbols)")
    ax_cons.set_xlabel("In-phase")
    ax_cons.set_ylabel("Quadrature")
    ax_cons.grid(True)
    ax_cons.set_xlim(-5, 5)
    ax_cons.set_ylim(-5, 5)

    # Figure 1: Equalizers (magitude dB + phase) stacked per i
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

    # Figure 2: Signals (x1||x2 and y real/imag) stacked per i (two columns)
    fig_sig, axes_sig = plt.subplots(N_IDX, 2, figsize=(12, max(6, 1.2 * N_IDX)), num=f"Signals i=0..{N_IDX-1}")
    for idx in range(N_IDX):
        ax_x1 = axes_sig[idx, 0] if N_IDX > 1 else axes_sig[0]
        ax_y = axes_sig[idx, 1] if N_IDX > 1 else axes_sig[1]

        xvals = xs_concat[idx]
        if xvals is not None and getattr(xvals, "size", 0) > 0:
            t = np.arange(len(xvals))
            ax_x1.plot(t, xvals, '-', lw=1)
            ax_x1.set_xlim(0, len(xvals) - 1)
            # Draw up to MAX_LINES vertical lines for detected starts (first 20)
            x1_len = len(latest_x1_by_i[idx]) if isinstance(latest_x1_by_i[idx], np.ndarray) else 0
            MAX_LINES = 20
            for s in indexes_copy[:MAX_LINES]:
                if not isinstance(s, (int, np.integer)):
                    continue
                pos = x1_len + int(s) + CP
                end_pos = pos + 2047
                if pos < 0 or pos >= len(xvals):
                    continue
                ax_x1.axvline(pos, color='red', linestyle='--', linewidth=1.6, alpha=0.15, zorder=15)
                if 0 <= end_pos < len(xvals):
                    ax_x1.axvline(end_pos, color='magenta', linestyle='--', linewidth=1.2, alpha=0.15, zorder=14)
        else:
            ax_x1.text(0.5, 0.5, "no x1||x2", ha='center', va='center', transform=ax_x1.transAxes)
        ax_x1.set_title(f"i={idx} x1||x2")
        ax_x1.set_xlabel("sample")
        ax_x1.set_ylabel("amp")

        yvals = ys_last[idx]
        if yvals is not None and getattr(yvals, "size", 0) > 0:
            t_y = np.arange(len(yvals))
            ax_y.plot(t_y, np.real(yvals), '-', label='real', linewidth=1)
            ax_y.plot(t_y, np.imag(yvals), '-', label='imag', linewidth=1)
            ax_y.set_xlabel("sample")
            ax_y.set_ylabel("value")
            ax_y.legend(fontsize='small')
        else:
            ax_y.text(0.5, 0.5, "no y", ha='center', va='center', transform=ax_y.transAxes)
        ax_y.set_title(f"i={idx} y (real / imag)")

    plt.tight_layout()
    plt.show()

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
     try:
         p.terminate()
     except Exception as e:
         print(f"Error terminating PyAudio: {e}")
         print(f"Stopped. Dropped frames: {dropped_frames}")



