import numpy as np
from scipy.io import wavfile

# Specify paths
in_path = 'sound.wav'
out_path = 'sound.csv'

# Load WAV
samplerate, data = wavfile.read(in_path)

## Rescale
#rescale = np.copy(data)
#rescale[:,0] = data[:,0]/4096
#rescale[:,1] = data[:,0]/4096
#rescale_s16 = np.int16(rescale) * 10
#wavfile.write('rescale.wav', samplerate, rescale_s16)

# Write CSV
file = open(out_path, 'w')
num_samples = data.shape[0]
for i in range(num_samples-1):
    n = file.write(str(data[i,0])+',')
n = file.write(str(data[num_samples-1,0]))
file.close

#FIN