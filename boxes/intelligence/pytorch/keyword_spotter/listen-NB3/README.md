# Intelligence : pytroch : KWS : Listen
A demo of the NB3's ability to respond to keywords using a pre-trained model running in PyTorch (qnnpack backend).

# Steps
- Record from microphone
- Compute mel spectrograms on audio snippets
- Classify each spectrogram
- If command detected, send to Hindbrain