#!/usr/bin/env python3
import pyaudio

def list_audio_devices():
    p = pyaudio.PyAudio()
    print("Available Audio Devices:")
    print("=" * 60)
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"Device {i}: {info['name']}")
        print(f"  Max input channels: {info['maxInputChannels']}")
        print(f"  Max output channels: {info['maxOutputChannels']}")
        print(f"  Default sample rate: {info['defaultSampleRate']}")
        print(f"  Host API: {p.get_host_api_info_by_index(info['hostApi'])['name']}")
        print("-" * 40)
    
    # Get default devices
    try:
        default_input = p.get_default_input_device_info()
        print(f"\nDefault Input Device: {default_input['index']} - {default_input['name']}")
        print(f"  Max input channels: {default_input['maxInputChannels']}")
    except OSError:
        print("\nNo default input device found")
    
    try:
        default_output = p.get_default_output_device_info()
        print(f"Default Output Device: {default_output['index']} - {default_output['name']}")
    except OSError:
        print("No default output device found")
    
    p.terminate()

if __name__ == "__main__":
    list_audio_devices()