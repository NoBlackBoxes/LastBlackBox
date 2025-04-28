import time
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Set paths
model_path = "/home/kampff/NoBlackBoxes/LastBlackBox/boxes/intelligence/LiteRT/_tmp/mobilenet_v1_1.0_224.tflite"
labels_path = "/home/kampff/NoBlackBoxes/LastBlackBox/boxes/intelligence/LiteRT/_tmp/labels.txt"
image_path = "/home/kampff/NoBlackBoxes/LastBlackBox/boxes/intelligence/LiteRT/_tmp/grace_hopper.bmp"

# Helper Function
def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

# Create interpreter
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)

# check the type of the input tensor
floating_model = input_details[0]['dtype'] == np.float32

# NxHxWxC, H:1, W:2
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
img = Image.open(image_path).resize((width, height))

# Add N dim
input_data = np.expand_dims(img, axis=0)

# Normalise (if floating point model)
if floating_model:
    input_mean = np.mean(input_data[:])
    input_std = np.std(input_data[:])
    input_data = (np.float32(input_data) - input_mean) / input_std

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

start_time = time.time()
interpreter.invoke()
stop_time = time.time()

output_data = interpreter.get_tensor(output_details[0]['index'])
results = np.squeeze(output_data)

top_k = results.argsort()[-5:][::-1]
labels = load_labels(labels_path)
for i in top_k:
    if floating_model:
        print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
    else:
        print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))

print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))