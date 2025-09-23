import os
import time
import base64
import openai
from dotenv import load_dotenv
import NB3.Vision.camera as Camera
import NB3.Sound.microphone as Microphone
import NB3.Sound.speaker as Speaker
import NB3.Sound.utilities as Utilities

# Get user name
username = os.getlogin()

# Set base path
base_path = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/intelligence/llms/chatNB3"

# Load OpenAI API Key
load_dotenv(f"{base_path}/.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Open camera, start, and wait for it to settle
camera = Camera.Camera(width=800, height=600)
camera.start()
time.sleep(1.0)

# Specify params
output_device = 1
num_channels = 2
sample_rate = 48000
buffer_size = int(sample_rate / 10)
max_samples = int(sample_rate * 10)

# List available sound devices
Utilities.list_devices()

# Initialize speaker
speaker = Speaker.Speaker(output_device, num_channels, 'int32', sample_rate, buffer_size)
speaker.start()

# Specify model
client = openai.OpenAI(api_key=OPENAI_API_KEY)
VISION_MODEL = "gpt-4o-mini"

# Acquire Frame (jpeg)
frame = camera.mjpeg()


# Encode frame (JPEG bytes) into a data URI
img_b64 = base64.b64encode(frame).decode("utf-8")
img_data_uri = f"data:image/jpeg;base64,{img_b64}"

# Ask a question about the image
question = "What do you see in this picture?"

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


## Play WAV file
#speaker.play_wav(wav_path)
#
## Wait for finish
#while speaker.is_playing():
#    time.sleep(0.1)

# Shutdown camera
camera.stop()

# Shutdown speaker
speaker.stop()
