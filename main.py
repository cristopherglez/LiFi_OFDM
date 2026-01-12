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

# Reset receiver state to ensure clean start
receiver.start_flag = False
receiver.start_index = 0
receiver.i = 0
receiver.sfo = 0
receiver.normalized_sfo = 0
receiver.sto = 0.0
receiver.sto_acc = 0.0
receiver.Eq = np.zeros(receiver.Nsub, dtype=complex)
receiver.y = np.array([], dtype=complex)
receiver.sfo_deviation = 0.0
receiver.sto_correction = -int((receiver.cp_length - 1) * receiver.oversampling_factor)
receiver.minn_value = 0.0
receiver.sto_counter = 0.0
print("Receiver state reset for clean start")

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
# Dynamic storage for all iterations (no hardcoded limit)
latest_eqs = {}                   # Eq per i (dict for dynamic sizing)
latest_x1_by_i = {}               # store last x1 (previous chunk) per i
latest_x2_by_i = {}               # store last x2 (current chunk) per i
latest_start_by_i = {}            # store detected start_index relative to x2 per i
latest_y_by_i = {}                # store last y per i
max_i_seen = -1                   # track highest iteration number seen
# New: global list of detected start indices (use this for vertical lines)
indexes = []                      # will be appended with start_index values (int)
# keep for backward compatibility (optional)
latest_eq = None                 # keep for backward compatibility (optional)
eq_ready = False                 # set True when final equalizer (i==5) has been captured
eq_plotted = False               # set True after we've plotted the equalizer once

# NEW: collect fewer y-vectors of expected length (Lfft/2 - 1)
expected_sym_len = Lfft // 2 - 1
collected_ys = []                # list to hold y arrays
symbols_needed = min(20, cfg['data_frame_length'])  # Don't request more symbols than available in data frame
symbols_collected = False


global vpos
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Initialize PyAudio and find a suitable input device
p = pyaudio.PyAudio()

# Function to find and test audio devices
def find_working_audio_device():
    print("Searching for working audio input device...")
    
    # Try default input device first
    try:
        default_input = p.get_default_input_device_info()
        device_index = default_input['index']
        max_channels = default_input['maxInputChannels']
        
        if max_channels >= 1:
            print(f"Trying default input device {device_index}: {default_input['name']} (max channels: {max_channels})")
            # Test if we can open it
            test_stream = p.open(format=FORMAT,
                               channels=1,
                               rate=RATE,
                               input=True,
                               input_device_index=device_index,
                               frames_per_buffer=1024)
            test_stream.close()
            print(f"✓ Default device {device_index} works!")
            return device_index, 1
    except Exception as e:
        print(f"✗ Default input device failed: {e}")
    
    # Try all available devices
    for i in range(p.get_device_count()):
        try:
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] >= 1:
                print(f"Trying device {i}: {info['name']} (max input channels: {info['maxInputChannels']})")
                
                # Test mono first
                try:
                    test_stream = p.open(format=FORMAT,
                                       channels=1,
                                       rate=RATE,
                                       input=True,
                                       input_device_index=i,
                                       frames_per_buffer=1024)
                    test_stream.close()
                    print(f"✓ Device {i} works with mono!")
                    return i, 1
                except Exception as e:
                    print(f"  ✗ Mono failed: {e}")
                
                # If mono fails, try stereo
                if info['maxInputChannels'] >= 2:
                    try:
                        test_stream = p.open(format=FORMAT,
                                           channels=2,
                                           rate=RATE,
                                           input=True,
                                           input_device_index=i,
                                           frames_per_buffer=1024)
                        test_stream.close()
                        print(f"✓ Device {i} works with stereo!")
                        return i, 2
                    except Exception as e:
                        print(f"  ✗ Stereo failed: {e}")
        except Exception as e:
            print(f"✗ Device {i} failed: {e}")
            continue
    
    raise RuntimeError("No working audio input device found!")

# Find working device
DEVICE_INDEX, CHANNELS = find_working_audio_device()
print(f"Using audio device {DEVICE_INDEX} with {CHANNELS} channels")

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
            
            # Handle stereo vs mono
            if CHANNELS == 2:
                # Take only the first channel for mono processing
                buf = buf[::2]  # Take every other sample (left channel)
            elif CHANNELS > 2:
                # Take first channel from multi-channel audio
                buf = buf.reshape(-1, CHANNELS)[:, 0]
                
            # Ensure we have the expected number of samples
            if len(buf) != CHUNK:
                buf = np.resize(buf, CHUNK)
                
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
    global collected_ys, symbols_collected, symbols_needed, indexes, max_i_seen
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

        # Capture latest x1, x2, start_index and y and Eq for this i (only when we have valid data)
        if i >= 0:
            with eq_lock:
                # Track the maximum iteration seen
                max_i_seen = max(max_i_seen, i)
                
                # Only store x1, x2 when we have a packet detection or valid processing
                # This prevents storing noise when there's no signal
                should_store_signal = (start_flag or 
                                     (isinstance(y, np.ndarray) and y.size > 0) or
                                     (Eq is not None and getattr(Eq, "size", 0) > 0))
                
                if should_store_signal:
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
                    # store per-i for this specific iteration
                    latest_start_by_i[i] = int(start_index)
                    # Print for iterations 0 through 25
                    if i <= 25:
                        print(f"Iteration i={i}: Detected start_index = {int(start_index)}")
                elif start_flag:
                    # Still update even if start_index is not valid, for completeness
                    latest_start_by_i[i] = start_index
                    # Print for iterations 0 through 25
                    if i <= 25:
                        print(f"Iteration i={i}: start_flag=True but start_index={start_index} (invalid)")

                if isinstance(y, np.ndarray) and y.size > 0:
                    latest_y_by_i[i] = y.copy()
                if Eq is not None and getattr(Eq, "size", 0) > 0:
                    latest_eqs[i] = Eq.copy()
                    if i == cfg['lts_repetitions']:
                        latest_eq = Eq.copy()
                        eq_ready = True
                        try:
                            vpos = receiver.start_index
                        except Exception:
                            vpos = None
        # If final equalizer has been seen, collect up to symbols_needed y-vectors (each of expected_sym_len)
        # Collect symbols from data frame processing phase (after LTS repetitions, but skip the transition iteration)
        if (eq_ready and (not symbols_collected) and isinstance(y, np.ndarray) and 
            y.size == expected_sym_len and i > cfg['lts_repetitions'] and 
            i <= cfg['lts_repetitions'] + cfg['data_frame_length']):
            with eq_lock:
                if len(collected_ys) < symbols_needed:
                    collected_ys.append(y.copy())
                    print(f"Collected symbol {len(collected_ys)}/{symbols_needed} from iteration i={i} (y.size={y.size})")
                    if len(collected_ys) >= symbols_needed:
                        symbols_collected = True
                        print(f"Finished collecting {symbols_needed} symbols")
        elif eq_ready and isinstance(y, np.ndarray) and y.size != expected_sym_len:
            print(f"Skipped i={i}: wrong y.size={y.size}, expected={expected_sym_len}")

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
        eqs_copy = dict((k, v.copy() if v is not None else None) for k, v in latest_eqs.items())
        # build concatenated x1||x2 per i (no start/end arrays here)
        xs_concat = {}
        for idx in latest_x1_by_i.keys() | latest_x2_by_i.keys():
            x1c = latest_x1_by_i.get(idx)
            x2c = latest_x2_by_i.get(idx)
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
            xs_concat[idx] = concat
        # copy the global indexes for plotting (may be empty)
        indexes_copy = list(indexes)
        ys_last = dict((k, v.copy() if isinstance(v, np.ndarray) else None) for k, v in latest_y_by_i.items())
        max_i_copy = max_i_seen

    # Plot constellation with different colors for each iteration
    fig_cons = plt.figure("Collected Constellation", figsize=(8, 8))
    ax_cons = fig_cons.add_subplot(111)
    
    if len(ys_copy_list) > 0:
        # Define a color map for different iterations using blue tones
        import matplotlib.cm as cm
        colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(ys_copy_list)))  # Use Blues colormap from light to dark blue
        
        all_real_vals = []
        all_imag_vals = []
        
        # Plot each iteration's symbols with different colors
        for iteration_idx, y_data in enumerate(ys_copy_list):
            # Filter out exactly 0+0j values before plotting
            non_zero_mask = (y_data != 0+0j)
            filtered_y = y_data[non_zero_mask]
            
            if filtered_y.size > 0:
                real_vals = np.real(filtered_y)
                imag_vals = np.imag(filtered_y)
                
                # Keep track of all values for axis limits
                all_real_vals.extend(real_vals)
                all_imag_vals.extend(imag_vals)
                
                # Plot with different color for each iteration
                ax_cons.scatter(real_vals, imag_vals, s=40, alpha=0.7, 
                              c=[colors[iteration_idx]], edgecolors='black', linewidth=0.3,
                              label=f'Iteration {iteration_idx + 1}')
        
        if all_real_vals and all_imag_vals:
            # Print debug info about the raw constellation values
            print(f"Raw constellation statistics ({len(all_real_vals)} total points across {len(ys_copy_list)} iterations):")
            print(f"  Real: min={np.min(all_real_vals):.2f}, max={np.max(all_real_vals):.2f}, mean={np.mean(all_real_vals):.2f}")
            print(f"  Imag: min={np.min(all_imag_vals):.2f}, max={np.max(all_imag_vals):.2f}, mean={np.mean(all_imag_vals):.2f}")
            
            # Set axis limits based on actual raw data range with some padding
            real_range = np.max(all_real_vals) - np.min(all_real_vals)
            imag_range = np.max(all_imag_vals) - np.min(all_imag_vals)
            padding = max(real_range, imag_range) * 0.1
            
            ax_cons.set_xlim(np.min(all_real_vals) - padding, np.max(all_real_vals) + padding)
            ax_cons.set_ylim(np.min(all_imag_vals) - padding, np.max(all_imag_vals) + padding)
            
            # Add legend if we have multiple iterations (limit legend entries to avoid clutter)
            if len(ys_copy_list) > 1:
                if len(ys_copy_list) <= 10:
                    ax_cons.legend(fontsize='small', loc='upper right')
                else:
                    # For many iterations, just show first and last in legend
                    handles, labels = ax_cons.get_legend_handles_labels()
                    selected_handles = [handles[0], handles[-1]] if len(handles) >= 2 else handles
                    selected_labels = [labels[0], labels[-1]] if len(labels) >= 2 else labels
                    ax_cons.legend(selected_handles, selected_labels, fontsize='small', loc='upper right')
        else:
            ax_cons.text(0.5, 0.5, "no data (all zeros filtered)", ha='center', va='center', transform=ax_cons.transAxes)
            ax_cons.set_xlim(-2, 2)
            ax_cons.set_ylim(-2, 2)
    else:
        ax_cons.text(0.5, 0.5, "no data", ha='center', va='center', transform=ax_cons.transAxes)
        ax_cons.set_xlim(-2, 2)
        ax_cons.set_ylim(-2, 2)
    
    ax_cons.set_title(f"Constellation by Iteration ({len(ys_copy_list)} iterations, {sum(len(y) for y in ys_copy_list)} total symbols)")
    ax_cons.set_xlabel("In-phase")
    ax_cons.set_ylabel("Quadrature")
    ax_cons.grid(True, alpha=0.3)
    ax_cons.set_aspect('equal')

    # Create multiple figures for equalizers and signals, up to 10 iterations per window
    if max_i_copy >= 0:
        all_iterations = sorted([i for i in eqs_copy.keys() if i >= 0])
        iterations_per_window = 10
        
        # Calculate number of windows needed
        num_windows = (len(all_iterations) + iterations_per_window - 1) // iterations_per_window
        
        # Plot Equalizers in separate windows
        for window_idx in range(num_windows):
            start_i = window_idx * iterations_per_window
            end_i = min(start_i + iterations_per_window, len(all_iterations))
            window_iterations = all_iterations[start_i:end_i]
            
            if not window_iterations:
                continue
                
            fig_eq, axes_eq = plt.subplots(len(window_iterations), 1, 
                                         figsize=(10, max(6, 0.6 * len(window_iterations))), 
                                         num=f"Equalizers Window {window_idx + 1} (i={window_iterations[0]}..{window_iterations[-1]})")
            
            for plot_idx, i in enumerate(window_iterations):
                ax_eq = axes_eq[plot_idx] if len(window_iterations) > 1 else axes_eq
                eq = eqs_copy.get(i)
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
                ax_eq.set_title(f"i={i} EQ")
            
            plt.tight_layout()

        # Plot Signals in separate windows  
        for window_idx in range(num_windows):
            start_i = window_idx * iterations_per_window
            end_i = min(start_i + iterations_per_window, len(all_iterations))
            window_iterations = all_iterations[start_i:end_i]
            
            if not window_iterations:
                continue
                
            fig_sig, axes_sig = plt.subplots(len(window_iterations), 2, 
                                           figsize=(12, max(6, 1.2 * len(window_iterations))), 
                                           num=f"Signals Window {window_idx + 1} (i={window_iterations[0]}..{window_iterations[-1]})")
            
            for plot_idx, i in enumerate(window_iterations):
                ax_x1 = axes_sig[plot_idx, 0] if len(window_iterations) > 1 else axes_sig[0]
                ax_y = axes_sig[plot_idx, 1] if len(window_iterations) > 1 else axes_sig[1]

                xvals = xs_concat.get(i)
                if xvals is not None and getattr(xvals, "size", 0) > 0:
                    t = np.arange(len(xvals))
                    ax_x1.plot(t, xvals, '-', lw=1)
                    ax_x1.set_xlim(0, len(xvals) - 1)
                    
                    # Plot vertical line for start index detected for this specific iteration (i)
                    x1_len = len(latest_x1_by_i.get(i, [])) if isinstance(latest_x1_by_i.get(i), np.ndarray) else 0
                    
                    # First, plot the specific start index for this iteration if available
                    start_idx_for_this_i = latest_start_by_i.get(i)
                    if start_idx_for_this_i is not None and isinstance(start_idx_for_this_i, (int, np.integer)):
                        pos = int(start_idx_for_this_i)
                        if 0 <= pos < len(xvals):
                            ax_x1.axvline(pos, color='red', linestyle='-', linewidth=2.5, alpha=0.8, 
                                         label=f'Start idx={start_idx_for_this_i}', zorder=20)
                            #Plot a second line at lfft*oversampling_factor after start index
                            second_pos = pos + receiver.Lfft * receiver.oversampling_factor
                            if 0 <= second_pos < len(xvals):
                                ax_x1.axvline(second_pos, color='orange', linestyle='--', linewidth=2.0, alpha=0.7, 
                                             label=f'Start+Lfft*OSF', zorder=15)
                    
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
                        ax_x1.axvline(pos, color='purple', linestyle='--', linewidth=1.0, 
                                     alpha=alpha_val, zorder=10)
                    
                    # Add legend if there's a main start line
                    if start_idx_for_this_i is not None:
                        ax_x1.legend(fontsize='small', loc='upper right')
                else:
                    ax_x1.text(0.5, 0.5, "no x1||x2", ha='center', va='center', transform=ax_x1.transAxes)
                ax_x1.set_title(f"i={i} x1||x2")
                ax_x1.set_xlabel("sample")
                ax_x1.set_ylabel("amp")

                yvals = ys_last.get(i)
                if yvals is not None and getattr(yvals, "size", 0) > 0:
                    t_y = np.arange(len(yvals))
                    ax_y.plot(t_y, np.real(yvals), '-', linewidth=1)
                    ax_y.plot(t_y, np.imag(yvals), '-', linewidth=1)
                    ax_y.set_xlabel("sample")
                    ax_y.set_ylabel("value")
                else:
                    ax_y.text(0.5, 0.5, "no y", ha='center', va='center', transform=ax_y.transAxes)
                ax_y.set_title(f"i={i} y (real / imag)")
            
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



