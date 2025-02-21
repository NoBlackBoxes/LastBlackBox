# Intelligence : NPU : Listen
A demo of the NB3's ability to respond to keywords using a pre-trained model running on the NPU (Coral EdgeTPU).

# Steps
- Record from microphone
- Compute mel spectrograms on audio snippets
- Classify each spectrogram
- If command detected, send to Hindbrain