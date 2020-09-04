import numpy as np
from scipy.io import wavfile

samplerate, data = wavfile.read('/home/kampff/test.wav')
rescale = np.copy(data)
rescale[:,0] = data[:,0]/4096
rescale[:,1] = data[:,0]/4096
rescale_s16 = np.int16(rescale) * 10
wavfile.write('/home/kampff/out.wav', samplerate, rescale_s16)
