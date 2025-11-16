import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from config import get_config

# Load config for audio parameters
cfg = get_config()
RATE =44100
DURATION = 10.0  # seconds
CHUNK = 1024

# Calculate total samples needed
total_samples = int(RATE * DURATION)
FORMAT = pyaudio.paInt16
CHANNELS = 1
DEVICE_INDEX = 0  # use default input

print(f"Recording {DURATION} seconds of audio at {RATE} Hz...")

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=DEVICE_INDEX,
                frames_per_buffer=CHUNK)

# Record audio
audio_data = []
samples_recorded = 0

try:
    while samples_recorded < total_samples:
        remaining = total_samples - samples_recorded
        chunk_size = min(CHUNK, remaining)
        
        data = stream.read(chunk_size, exception_on_overflow=False)
        buf = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        audio_data.extend(buf)
        samples_recorded += len(buf)
        
        # Progress indicator
        progress = samples_recorded / total_samples * 100
        print(f"\rRecording: {progress:.1f}%", end="", flush=True)

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()

print(f"\nRecording complete. Recorded {len(audio_data)} samples.")

# Convert to numpy array and create time axis
audio_array = np.array(audio_data)
time_axis = np.arange(len(audio_array)) / RATE

# Plot the raw audio
plt.figure("Raw Audio Recording", figsize=(12, 6))
plt.plot(time_axis, audio_array, linewidth=0.5)
plt.title(f"Raw Audio Recording - {DURATION} seconds")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)
plt.xlim(0, DURATION)

# Add some statistics
mean_val = np.mean(audio_array)
std_val = np.std(audio_array)
max_val = np.max(np.abs(audio_array))

plt.text(0.02, 0.98, f"Mean: {mean_val:.1f}\nStd: {std_val:.1f}\nMax: {max_val:.1f}", 
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

print(f"Audio statistics:")
print(f"  Duration: {len(audio_array)/RATE:.2f} seconds")
print(f"  Sample rate: {RATE} Hz")
print(f"  Mean: {mean_val:.2f}")
print(f"  Standard deviation: {std_val:.2f}")
print(f"  Max amplitude: {max_val:.2f}")
