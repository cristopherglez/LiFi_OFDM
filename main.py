import numpy as np
import pyaudio
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from config import get_config
from vlc_receiver import OFDMReceiver

# Get configuration
cfg = get_config()

# Initialize queues
audio_queue = queue.Queue(maxsize=cfg['queue_size'])
result_queue = queue.Queue(maxsize=cfg['queue_size'])

# Initialize receiver with correct parameters
receiver = OFDMReceiver(
    Lfft=cfg['Lfft'],
    cp_length=cfg['cp_length'],
    data_frame_length=cfg['data_frame_length'],
    lts_repetitions=cfg['lts_repetitions'],
    sfo_repetitions=cfg['sfo_repetitions'],
    sts_no_cp=cfg['sts_no_cp'],
    lts_no_cp=cfg['lts_no_cp'],
    oversampling_factor=cfg['oversampling_factor']
)

# Control flags
running = False
constellation = np.array([], dtype=complex)
debug_count = 0
debug_info = []
packet_detected = False

def audio_callback(in_data, frame_count, time_info, status):
    """Audio input callback"""
    try:
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        audio_data = audio_data / 32768.0
        audio_queue.put_nowait(audio_data)
    except queue.Full:
        pass
    return (None, pyaudio.paContinue)

def processing_thread():
    """Main processing thread"""
    global debug_count, debug_info, packet_detected
    buffer = np.array([], dtype=np.float32)
    
    while running:
        try:
            chunk = audio_queue.get(timeout=0.1)
            buffer = np.concatenate([buffer, chunk])
            
            if len(buffer) >= 2 * cfg['chunk_size']:
                x1 = buffer[:cfg['chunk_size']]
                x2 = buffer[cfg['chunk_size']:2*cfg['chunk_size']]
                buffer = buffer[cfg['chunk_size']:]
                
                # Process with receiver - handle correct return values
                start_flag, start_idx, y, i, Eq = receiver.process(x1, x2)
                
                # Check if packet detected for the first time
                if start_flag and not packet_detected:
                    packet_detected = True
                    print("Packet detected! Beginning debug recording...")
                
                # Store debug info for first 10 iterations ONLY after packet detection
                if packet_detected and debug_count < 10:
                    combined_signal = np.concatenate([x1, x2])
                    
                    # Handle different types of y and Eq
                    y_copy = y
                    if hasattr(y, 'copy') and callable(getattr(y, 'copy')):
                        y_copy = y.copy()
                    elif isinstance(y, dict):
                        y_copy = y.copy() if y else {}
                    
                    Eq_copy = Eq
                    if hasattr(Eq, 'copy') and callable(getattr(Eq, 'copy')):
                        Eq_copy = Eq.copy()
                    
                    debug_info.append({
                        'i': i,
                        'combined_signal': combined_signal,
                        'y': y_copy,
                        'Eq': Eq_copy,
                        'start_flag': start_flag,
                        'start_idx': start_idx
                    })
                    debug_count += 1
                    print(f"Debug {debug_count}: i={i}, start_flag={start_flag}, start_idx={start_idx}")
                    print(f"  y type: {type(y)}")
                    print(f"  Eq type: {type(Eq)}")
                
                # Only send results to visualization if packet detected
                if packet_detected:
                    result = {
                        'start_flag': start_flag,
                        'start_idx': start_idx,
                        'y': y,
                        'i': i,
                        'Eq': Eq,
                    }
                    
                    try:
                        result_queue.put_nowait(result)
                    except queue.Full:
                        pass
                        
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Processing error: {e}")
            import traceback
            traceback.print_exc()

def update_plot(frame):
    """Update visualization"""
    global constellation, debug_info, packet_detected
    
    # Only update plots if packet has been detected
    if not packet_detected:
        plt.clf()
        plt.text(0.5, 0.5, 'Waiting for packet...', 
                horizontalalignment='center', verticalalignment='center', 
                transform=plt.gca().transAxes, fontsize=16)
        return
    
    try:
        result = result_queue.get_nowait()
        
        # Update constellation if we have symbols
        if hasattr(result['y'], '__len__') and len(result['y']) > 0:
            constellation = result['y']
        
        plt.clf()
        
        # Main constellation plot
        plt.subplot(2, 2, 1)
        if len(constellation) > 0:
            plt.scatter(np.real(constellation), np.imag(constellation), alpha=0.6)
            plt.title(f"Constellation - i={result['i']}")
            plt.xlabel("Real")
            plt.ylabel("Imaginary")
            plt.grid(True)
            plt.axis('equal')
        else:
            plt.title(f"Constellation - i={result['i']} (No data)")
        
        # Debug info plots for first 10 iterations
        if len(debug_info) > 0:
            last_debug = debug_info[-1]
            
            # Plot last debug y
            plt.subplot(2, 2, 2)
            if hasattr(last_debug['y'], '__len__') and len(last_debug['y']) > 0:
                plt.scatter(np.real(last_debug['y']), np.imag(last_debug['y']), 
                           alpha=0.6, color='red')
                plt.title(f"Debug y (i={last_debug['i']})")
                plt.xlabel("Real")
                plt.ylabel("Imaginary")
                plt.grid(True)
                plt.axis('equal')
            else:
                plt.title(f"Debug y (i={last_debug['i']}) - No data")
            
            # Plot Eq
            plt.subplot(2, 2, 3)
            if hasattr(last_debug['Eq'], '__len__') and len(last_debug['Eq']) > 0:
                plt.plot(np.abs(last_debug['Eq']), 'b-', label='|Eq|')
                plt.title(f"Equalizer |Eq| (i={last_debug['i']})")
                plt.xlabel("Subcarrier")
                plt.ylabel("Magnitude")
                plt.grid(True)
                plt.legend()
            else:
                plt.title(f"Equalizer (i={last_debug['i']}) - No data")
            
            # Plot combined signal spectrum
            plt.subplot(2, 2, 4)
            if len(last_debug['combined_signal']) > 0:
                spectrum = np.abs(np.fft.fft(last_debug['combined_signal']))
                plt.plot(spectrum[:len(spectrum)//2])
                plt.title(f"Signal Spectrum (i={last_debug['i']})")
                plt.xlabel("Frequency bin")
                plt.ylabel("Magnitude")
                plt.grid(True)
            else:
                plt.title(f"Signal Spectrum (i={last_debug['i']}) - No data")
        
    except queue.Empty:
        pass

def main():
    global running
    
    running = True
    
    # Initialize audio
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=cfg['channels'],
        rate=cfg['sample_rate'],
        input=True,
        frames_per_buffer=cfg['chunk_size'],
        stream_callback=audio_callback,
        input_device_index=cfg['audio_device']
    )
    
    # Start processing thread
    proc_thread = threading.Thread(target=processing_thread, daemon=True)
    proc_thread.start()
    
    # Start visualization
    fig = plt.figure(figsize=(12, 8))
    ani = FuncAnimation(fig, update_plot, interval=1000//cfg['visualization_update_rate'])
    
    try:
        print("Receiver started. Close plot window to stop.")
        plt.show()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        running = False
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()



