# Intelligence : pytroch : KWS : Listen
A demo of the NB3's ability to respond to keywords using a pre-trained model running in PyTorch (qnnpack backend).

## Steps
- Record from microphone
- Compute mel spectrograms on audio snippets
- Classify each spectrogram using the neural network
- If a "command" is detected, send to Hindbrain

## Setup
This demo uses PyTorch to run the quantized neural network directly on the Raspberry Pi.
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

