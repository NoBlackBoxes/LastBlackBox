# ChatNB3: Have a conversation with your NB3 (Google Gemini version)
import os, time, dotenv
import numpy as np
from google import genai
from google.genai import types
import LBB.config as Config
import importlib.util
if importlib.util.find_spec("picamera2") is not None:
    import NB3.Vision.camera as Camera                  # NB3 Camera
else:
    import NB3.Vision.webcam as Camera                  # Webcam (PC)
import NB3.Sound.speaker as Speaker
import NB3.Sound.utilities as Utilities

# Specify Paths
box_root = f"{Config.boxes_path}/intelligence"
project_root = f"{box_root}/LLMs/gemini/chatNB3"

# Load Gemini API Key
dotenv.load_dotenv(f"{Config.boxes_path}/intelligence/LLMs/gemini/.env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# List available sound devices
Utilities.list_devices()

# Get speaker device by name (NB3: "MAX", PC: select based on listed output devices)
output_device = Utilities.get_output_device_by_name("HD")
if output_device == -1:
    exit("Output device not found")

# Specify params
num_channels = 2
sample_rate = 48000
buffer_size = int(sample_rate / 10)
max_samples = int(sample_rate * 10)

# Initialize speaker
speaker = Speaker.Speaker(output_device, num_channels, 'int32', sample_rate, buffer_size)
speaker.start()

# Open camera, start, and wait for it to settle
camera = Camera.Camera(width=800, height=600)
camera.start()
time.sleep(1.0)

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key=GEMINI_API_KEY)

# Acquire Frame (jpeg)
frame = camera.capture(mjpeg=True)
image_part = types.Part.from_bytes(
    data=frame,
    mime_type='image/jpeg'
)

# Call the API
prompt = "Provide a concise and vivid description of this image. Focus on the main subjects and action. hey reply can only be a single short sentence."
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[prompt, image_part]
)
print(response.text.strip())

# Get TTS audio output
response = client.models.generate_content(
   model="gemini-2.5-flash-preview-tts",
   contents=response.text.strip(),
   config=types.GenerateContentConfig(
      response_modalities=["AUDIO"],
      speech_config=types.SpeechConfig(
         voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
               voice_name='Kore',
            )
         )
      ),
   )
)

# Convert bytes -> int16 -> float32 [-1, 1)
data = response.candidates[0].content.parts[0].inline_data.data
pcm16 = np.frombuffer(data, dtype=np.int16)
audio_f32 = pcm16.astype(np.float32) / 32768.0
audio_f32 = np.vstack((audio_f32, audio_f32)).T
audio_48k = np.repeat(audio_f32, 2, axis=0)
print(audio_48k.shape)
print(audio_48k)
speaker.volume = 0.5

# Play sound data
speaker.write(audio_48k)

# Wait for finish
while speaker.is_playing():
    time.sleep(0.1)
    print(f"{speaker.current_sample} of {speaker.max_samples}")

# Shutdown camera
camera.stop()

# Shutdown speaker
speaker.stop()

# FIN