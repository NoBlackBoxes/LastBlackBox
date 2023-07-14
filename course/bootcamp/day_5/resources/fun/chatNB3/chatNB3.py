import sys
import openai
import pyttsx3
import pyaudio
import wave
import numpy as np
import curses

# Set OpenAI API Key (secret!!!)
openai.api_key = "<secret>"
system_prompt = "You are small two wheeled robot shaped like a brain named NB3. Your task is to listen to snippets of audio from a neuroscience course and respond with witty comments and funny jokes.. Only produce short one or two sentence replies."

# Initialize conversation history
conversation = [
    {"role": "system", "content": "You are small two wheeled robot shaped like a brain. Your name is NB3, which stands for no black box bot. Your task is to respond to questions about neuroscience and technology, or anything really, with a short snarky but accurate reply."},
]

# Initialize speech engine
engine = pyttsx3.init()

# Set sound recording format
CHUNK = 1600                # Buffer size
FORMAT = pyaudio.paInt16    # Data type
CHANNELS = 1                # Number of channels
RATE = 16000                # Sample rate (Hz)
MAX_DURATION = 5            # Max recording duration
WAVE_OUTPUT_FILENAME = "speech.wav"

# Get pyaudio object
pya = pyaudio.PyAudio()

# Open audio stream (from default device)
stream = pya.open(format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            start=False,
            frames_per_buffer=CHUNK)

# Setup the curses screen window
screen = curses.initscr()
curses.noecho()
curses.cbreak()
screen.nodelay(True)
 
# --------------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------------

# Function to record speech snippets to a WAV file
def record_speech(stream):

    # Prepare a WAV file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(RATE)

    # Start streaming audio
    stream.start_stream()

    # Append frames of data until key (spacebar) is pressed
    frames = []
    for i in range(0, int(RATE / CHUNK * MAX_DURATION)):
        # Read raw data and append
        raw_data = stream.read(CHUNK)
        frames.append(raw_data)
    
        # Check for key press ('z')
        char = screen.getch()
        if char == ord('z'):
            break

    # Stop stream
    stream.stop_stream()

    # Write to WAV file
    wf.writeframes(b''.join(frames))
    
    # Close WAV file
    wf.close()

    return
# --------------------------------------------------------------------------------


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
        record_speech(stream)
        screen.erase()        

        # Get transcription from Whisper
        audio_file= open("speech.wav", "rb")
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

        # Speak reply
        engine.say(reply)
        engine.runAndWait()
        screen.addstr(8, 0, "NB3: {0}\n".format(reply), curses.A_NORMAL)
        screen.refresh()

finally:
    # shut down
    stream.close()
    pya.terminate()
    curses.nocbreak()
    screen.keypad(0)
    curses.echo()
    curses.endwin()
# FIN