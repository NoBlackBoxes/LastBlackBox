import os
import sys
import time
import openai
from dotenv import load_dotenv
from google.cloud import texttospeech
import numpy as np
import curses

# Locals libs
import libs.NBB_sound as sound

# Reimport
import importlib
importlib.reload(sound)

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/LastBlackBox'
box_path = repo_path + '/boxes/intelligence/llms/chatNB3'
input_wav_path = box_path + '/_tmp/input.wav'
output_wav_path = box_path + '/_tmp/output.wav'
output_mp3_path = box_path + '/_tmp/output.mp3'

# Indicate OpenAI API Key environmental variable
load_dotenv("_tmp/.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize conversation history
conversation = [
    {"role": "system", "content": "You are small two wheeled robot shaped like a brain. Your name is NB3, which stands for no black box bot. Your task is to respond to questions about neuroscience and technology, or anything really, with a short snarky but accurate reply."},
]

# Instantiates a Google TTS client
google_tts_client = texttospeech.TextToSpeechClient()

# Specify params
input_device = 0
output_device = 0
num_input_channels = 2
num_output_channels = 1
sample_rate = 44100
buffer_size = int(sample_rate / 100)
buffer_size = int(sample_rate / 100)
max_samples = int(sample_rate * 10)

# List available sound devices
sound.list_devices()

# Initialize microphone
microphone = sound.microphone(input_device, num_input_channels, 'int32', sample_rate, buffer_size, max_samples)
microphone.start()

# Initialize speaker
speaker = sound.speaker(output_device, num_output_channels, 'int16', sample_rate, buffer_size)
speaker.start()

# Clear error ALSA/JACK messages from terminal
os.system('cls' if os.name == 'nt' else 'clear')

# Setup the curses screen window
screen = curses.initscr()
curses.noecho()
curses.cbreak()
screen.nodelay(True)
 
# --------------------------------------------------------------------------------
# Chat Loop
# --------------------------------------------------------------------------------
try:
    while True:

        # Wait to start talking
        screen.addstr(0, 0, "Press 'z' to talk to your NB3 ('q' to quit):")
        screen.clrtoeol()
        while True:
            char = screen.getch()
            if char == ord('q'):
                sys.exit()
            elif char == ord('z'):
                break

        # Start recording
        screen.addstr("...press 'z' again to stop speaking.", curses.A_UNDERLINE)
        microphone.reset()
        while True:
            char = screen.getch()
            if char == ord('q'):
                sys.exit()
            elif char == ord('z'):
                break
        screen.erase()        
        microphone.save_wav(input_wav_path, max_samples)

        # Get transcription from Whisper
        audio_file= open(input_wav_path, "rb")
        transcription = openai.Audio.transcribe("whisper-1", audio_file)['text']
        conversation.append({'role': 'user', 'content': f'{transcription}'})
        screen.addstr(4, 0, "You: {0}\n".format(transcription), curses.A_STANDOUT)
        screen.refresh()

        # Get ChatGPT response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.2,
            messages=conversation
        )

        # Extract and display reply
        reply = response['choices'][0]['message']['content']
        conversation.append({'role': 'assistant', 'content': f'{reply}'})

        # Synthesize reply
        synthesis_input = texttospeech.SynthesisInput(text=reply)
        voice = texttospeech.VoiceSelectionParams(language_code="en-GB", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16, sample_rate_hertz=44100)
        response = google_tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        with open(output_wav_path, "wb") as out:
            out.write(response.audio_content)

        # Speak reply
        speaker.play_wav(output_wav_path)
        screen.addstr(8, 0, "NB3: {0}\n".format(reply), curses.A_NORMAL)
        while speaker.is_playing():
            time.sleep(0.1)
        screen.refresh()

finally:
    # Shutdown microphone
    microphone.stop()

    # Shutdown speaker
    speaker.stop()

    # Shutdown curses
    curses.nocbreak()
    screen.keypad(0)
    curses.echo()
    curses.endwin()
# FIN