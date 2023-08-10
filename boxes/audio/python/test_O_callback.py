import os
import wave
import time
import pyaudio

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/LastBlackBox'
box_path = repo_path + '/boxes/audio'
wav_path = box_path + '/_data/sounds/Bach_prelude_C_major.wav'

with wave.open(wav_path, 'rb') as wf:

    # Define callback for playback (1)
    def callback(in_data, frame_count, time_info, status):
        data = wf.readframes(frame_count)
        # If len(data) is less than requested frame_count, PyAudio automatically
        # assumes the stream is finished, and the stream stops.
        return (data, pyaudio.paContinue)

    # Instantiate PyAudio and initialize PortAudio system resources (2)
    p = pyaudio.PyAudio()

    # Open stream using callback (3)
    stream = p.open(output_device_index=1,
                    format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    stream_callback=callback)

    # Wait for stream to finish (4)
    while stream.is_active():
        time.sleep(0.1)

    # Close the stream (5)
    stream.close()

    # Release PortAudio system resources (6)
    p.terminate()