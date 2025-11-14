#!/usr/bin/env python3
"""
Audio Device Detection Script
This script lists all available audio input/output devices using PyAudio.
Use this to find the correct device index for your audio application.
"""

import pyaudio
import sys

def list_audio_devices():
    """List all available audio devices with their properties."""
    
    try:
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        print("PyAudio Audio Device Detection")
        print("=" * 60)
        print(f"Total devices found: {p.get_device_count()}")
        print("=" * 60)
        
        # List all audio devices
        input_devices = []
        output_devices = []
        
        for i in range(p.get_device_count()):
            try:
                info = p.get_device_info_by_index(i)
                
                print(f"\nDevice {i}: {info['name']}")
                print(f"  Host API: {info['hostApi']}")
                print(f"  Max input channels: {info['maxInputChannels']}")
                print(f"  Max output channels: {info['maxOutputChannels']}")
                print(f"  Default sample rate: {info['defaultSampleRate']:.0f} Hz")
                
                # Categorize devices
                if info['maxInputChannels'] > 0:
                    input_devices.append((i, info['name'], info['maxInputChannels']))
                    print(f"  ✓ INPUT CAPABLE")
                
                if info['maxOutputChannels'] > 0:
                    output_devices.append((i, info['name'], info['maxOutputChannels']))
                    print(f"  ✓ OUTPUT CAPABLE")
                
                print("-" * 50)
                
            except Exception as e:
                print(f"  Error reading device {i}: {e}")
        
        # Show summary of input devices
        print("\n" + "=" * 60)
        print("INPUT DEVICES SUMMARY:")
        print("=" * 60)
        if input_devices:
            for idx, name, channels in input_devices:
                print(f"Device {idx}: {name} ({channels} channels)")
        else:
            print("No input devices found!")
        
        # Show summary of output devices
        print("\n" + "=" * 60)
        print("OUTPUT DEVICES SUMMARY:")
        print("=" * 60)
        if output_devices:
            for idx, name, channels in output_devices:
                print(f"Device {idx}: {name} ({channels} channels)")
        else:
            print("No output devices found!")
        
        # Get and display default devices
        print("\n" + "=" * 60)
        print("DEFAULT DEVICES:")
        print("=" * 60)
        
        try:
            default_input = p.get_default_input_device_info()
            print(f"Default INPUT device: {default_input['index']} - {default_input['name']}")
            print(f"  Channels: {default_input['maxInputChannels']}")
            print(f"  Sample rate: {default_input['defaultSampleRate']:.0f} Hz")
        except Exception as e:
            print(f"No default input device: {e}")
        
        try:
            default_output = p.get_default_output_device_info()
            print(f"Default OUTPUT device: {default_output['index']} - {default_output['name']}")
            print(f"  Channels: {default_output['maxOutputChannels']}")
            print(f"  Sample rate: {default_output['defaultSampleRate']:.0f} Hz")
        except Exception as e:
            print(f"No default output device: {e}")
        
        # Recommendations for main.py
        print("\n" + "=" * 60)
        print("RECOMMENDATIONS FOR main.py:")
        print("=" * 60)
        
        if input_devices:
            # Find the best input device (prefer default, then first available)
            try:
                default_input = p.get_default_input_device_info()
                recommended_idx = default_input['index']
                recommended_name = default_input['name']
                print(f"Recommended DEVICE_INDEX = {recommended_idx}  # {recommended_name}")
            except:
                # Use first available input device
                recommended_idx, recommended_name, _ = input_devices[0]
                print(f"Recommended DEVICE_INDEX = {recommended_idx}  # {recommended_name}")
            
            print("Update your main.py file:")
            print(f"  DEVICE_INDEX = {recommended_idx}")
            print("  CHANNELS = 1")
        else:
            print("WARNING: No input devices found! Audio recording will not work.")
            print("Check your audio hardware and drivers.")
        
        # Cleanup
        p.terminate()
        
    except Exception as e:
        print(f"Error initializing PyAudio: {e}")
        print("\nPossible solutions:")
        print("1. Check if audio drivers are properly installed")
        print("2. Try: sudo apt install pulseaudio pulseaudio-utils")
        print("3. Restart your system")
        return False
    
    return True

def test_device(device_index, sample_rate=44100, channels=1):
    """Test if a specific device can be opened for recording."""
    
    print(f"\nTesting device {device_index} for recording...")
    print(f"Sample rate: {sample_rate} Hz, Channels: {channels}")
    
    try:
        p = pyaudio.PyAudio()
        
        # Try to open the device
        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=1024
        )
        
        print("✓ SUCCESS: Device can be opened for recording")
        
        # Try to read some data
        data = stream.read(1024, exception_on_overflow=False)
        print("✓ SUCCESS: Can read audio data from device")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        try:
            p.terminate()
        except:
            pass
        return False

if __name__ == "__main__":
    print("Starting audio device detection...\n")
    
    # List all devices
    success = list_audio_devices()
    
    if success and len(sys.argv) > 1:
        # Test specific device if provided as command line argument
        try:
            device_idx = int(sys.argv[1])
            test_device(device_idx)
        except ValueError:
            print(f"\nInvalid device index: {sys.argv[1]}")
        except Exception as e:
            print(f"\nError testing device: {e}")
    
    print(f"\nDevice detection complete!")
    print(f"Usage: python devices.py [device_index_to_test]")