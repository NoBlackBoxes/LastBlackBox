# ChatNB3

***Let's have a conversation with our NB3 (powered by Whisper, GPT, and TTS)***

### Microphone (NB3 Ears) ready?
Is at least one of your microphones working? If so, then you should be ready to go!

### Adding a speaker
You can connect any speaker with a standard 3.5 mm plug (the headphone plug) into the audio jack on your raspberry pi, directly next to where the camera connects.

<p align="center">
<img src="resources/images/3_5mm_plug.png" alt="3.5 mm Plug" width="200" height="200">
</p>

## Install required Python packages
```bash
pip3 install openai
pip3 install pyttsx3
```

## Install text-to-speech library
```bash
sudo apt install espeak
```

### OpenAI API Key
This is a special code that you can get from OpenAI that allows you to send requests to their computers running very large neural networks. Each time you make a request, they charge a very tiny amount of money. This is their business model. I will provide you with an OpenAI key to use during the course...but please don't share it with the internet. (It has a spending cap, but it will be quickly used up if the "world" gets access.)

