# Intelligence :  LLMs : ChatNB3

***Let's have a conversation with our NB3 (powered by Whisper, GPT, and TTS)***

### Microphone (NB3 Ears) ready?
Is at least one of your microphones working? If so, then you should be ready to go!

### Speaker (NB3 Mouth) ready?
Is your speaker working? If so, then you should be ready to go!

## Install required Python packages
```bash
pip install openai
pip install piper-tts
pip install --upgrade google-cloud-texttospeech
pip install python-dotenv
```

## Test text-to-speech (and download voice models)
```bash
mkdir _tmp
cd _tmp

# If pip install piper-tts fails, then you can donwload the binary here:
# wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz

# Download some voices (model and json)
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/it/it_IT/riccardo/x_low/it_IT-riccardo-x_low.onnx
wget https://huggingface.co/rhasspy/piper-voices/raw/v1.0.0/it/it_IT/riccardo/x_low/it_IT-riccardo-x_low.onnx.json

# Synthesize speech (to WAV file)
echo "Congratulazioni per aver vinto il robot d'oro!" | piper --model it_IT-riccardo-x_low.onnx --output_file ciao.wav

# Play it
aplay -D plughw:1 -c2 -r 16000 -f S16_LE -t wav -V stereo -v ciao.wav
```

### OpenAI API Key
This is a special code that you can get from OpenAI that allows you to send requests to their computers running very large neural networks. Each time you make a request, they charge a very tiny amount of money. This is their business model. I will provide you with an OpenAI key to use during the course...but please don't share it with the internet. (It has a spending cap, but it will be quickly used up if the "world" gets access.)

