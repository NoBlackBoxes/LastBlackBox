import os
import time
import argparse
import numpy as np
import soundfile as sf
import threading
import sys
import shutil
import termios
import tty
import select

# Local libs
import NB3.Sound.speaker as Speaker
import NB3.Sound.utilities as Utilities

# Reimport
import importlib
importlib.reload(Speaker)
importlib.reload(Utilities)

def print_progress_bar(elapsed, duration, bar_length=40):
    progress = min(elapsed / duration, 1.0)
    filled_length = int(bar_length * progress)
    bar = "#" * filled_length + "-" * (bar_length - filled_length)
    print(f"\r[{bar}] {elapsed:.1f}s / {duration:.1f}s", end='', flush=True)

def is_key_pressed():
    rlist, _, _ = select.select([sys.stdin], [], [], 0)
    return bool(rlist)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Play a WAV file with playback controls and profiling.')
    parser.add_argument('wav_path', type=str, help='Path to the WAV file to be played.')
    parser.add_argument(
        '--device', type=int, default=1,
        help='Output sound device ID. Use "NB3.Sound.utilities.list_devices()" to view available devices. Default: 1.'
    )
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.isfile(args.wav_path):
        print(f"Error: File '{args.wav_path}' not found.")
        return
    
    # Get file parameters
    try:
        with sf.SoundFile(args.wav_path) as file:
            sample_rate = file.samplerate
            num_channels = file.channels
            duration = len(file) / sample_rate
    except Exception as e:
        print(f"Error reading WAV file: {e}")
        return

    # Specify params
    output_device = args.device
    buffer_size = int(sample_rate / 10)

    # List available sound devices
    Utilities.list_devices()

    # Initialize speaker
    speaker = Speaker.Speaker(output_device, num_channels, 'int32', sample_rate, buffer_size)
    speaker.start()

    # Display playback details
    print(f"Playing: {args.wav_path}")
    print(f"Sample Rate: {sample_rate} Hz, Channels: {num_channels}, Duration: {duration:.2f} sec")
    print("Press 'q' to stop playback.")

    # Start playback
    speaker.play_wav(args.wav_path)
    
    # Track playback progress and listen for keypress
    start_time = time.time()
    stop_playback = False
    old_settings = termios.tcgetattr(sys.stdin.fileno())
    try:
        tty.setcbreak(sys.stdin.fileno())  # Set terminal to cbreak mode
        while speaker.is_playing() and not stop_playback:
            elapsed = time.time() - start_time
            print_progress_bar(elapsed, duration)
            time.sleep(1)
            if is_key_pressed():
                key = sys.stdin.read(1)
                if key.lower() == 'q':
                    stop_playback = True
                    speaker.stop()
                    print("\nPlayback stopped by user.")
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)  # Restore terminal settings

    print("\n")
    
    # Shutdown speaker
    speaker.stop()

    # Report profiling stats
    print("Profiling:")
    print(f"- Avg Callback Duration (us): {speaker.callback_accum / speaker.callback_count * 1000000.0:.2f}")
    print(f"- Max Callback Duration (us): {speaker.callback_max * 1000000.0:.2f}")

if __name__ == "__main__":
    main()
