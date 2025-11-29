import os, time, base64, openai
import numpy as np
import LBB.config as Config
import dotenv
import importlib.util
if importlib.util.find_spec("picamera2") is not None:
    import NB3.Vision.camera as Camera                  # NB3 Camera
else:
    import NB3.Vision.webcam as Camera                  # Webcam (PC)
import NB3.Sound.microphone as Microphone
import NB3.Sound.speaker as Speaker
import NB3.Sound.utilities as Utilities

# Specify Paths
box_root = f"{Config.boxes_path}/intelligence"
project_root = f"{box_root}/llms/chatNB3"

# Load OpenAI API Key
dotenv.load_dotenv(f"{project_root}/.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Open camera, start, and wait for it to settle
camera = Camera.Camera(width=800, height=600)
camera.start()
time.sleep(1.0)

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

# Specify model
client = openai.OpenAI(api_key=OPENAI_API_KEY)
VISION_MODEL = "gpt-4o-mini"

# Acquire Frame (jpeg)
frame = camera.capture(mjpeg=True)

# Encode frame (JPEG bytes) into a data URI
img_b64 = base64.b64encode(frame).decode("utf-8")
img_data_uri = f"data:image/jpeg;base64,{img_b64}"

# Ask a question about the image
question = "What do you see in this picture? Please give a one sentence reply."

resp = client.responses.create(
    model=VISION_MODEL,
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": question},
            {"type": "input_image", "image_url": img_data_uri}
        ]
    }]
)
print("Model reply:", resp.output_text)

# Turn the reply text into audio (WAV bytes)
VOICE = "shimmer"

tts = client.audio.speech.create(
    model="gpt-4o-mini-tts",        # text-to-speech model
    voice=VOICE,
    response_format="pcm",          # or "pcm" for raw 16-bit little-endian (24 kHz)
    input=resp.output_text,
)

# The SDK usually exposes .read() for the byte stream
pcm_bytes = tts.read() if hasattr(tts, "read") else bytes(tts)

# Convert bytes -> int16 -> float32 [-1, 1)
pcm16 = np.frombuffer(pcm_bytes, dtype=np.int16)
audio_f32 = pcm16.astype(np.float32) / 32768.0
audio_f32 = np.vstack((audio_f32, audio_f32)).T
audio_48k = np.repeat(audio_f32, 2, axis=0)
print(audio_48k.shape)
print(audio_48k)
speaker.volume = 0.75

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
