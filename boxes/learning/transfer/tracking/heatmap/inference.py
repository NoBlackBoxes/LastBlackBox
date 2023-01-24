import cv2
import torch
import model
import numpy as np
from torchvision import transforms

# Specify video or camera
live_capture = False
if live_capture:
    width = 640
    height = 480
else:
    width = 512
    height = 512

# Specify paths
repo_path = '/home/kampff/NoBlackBoxes/repos/OtherBlackBoxes'
box_path = repo_path + '/boxes/ai/tracking/nose/heatmap'
model_path = box_path + '/_tmp/custom.pt'
video_path = repo_path + '/boxes/ai/tracking/_data/nose.mp4'

# Load model
custom_model = model.custom()
custom_model.load_state_dict(torch.load(model_path))

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Move model to device
custom_model.to(device)

# Specify transforms for inputs
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Get video capture object
if live_capture:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create named window for diaply
cv2.namedWindow('tracking')

# Save?
save = True
video_path = box_path + '/_tmp/example.avi'
if save:
    # Open output video
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('F','M','P','4'), 30, (width, height))
    if not video.isOpened():
        print("Cannot open video file")
        exit()

# Loop until 'q' pressed
while(True):
    # Read most recent frame
    ret, frame = cap.read()

    # End of video?
    if (not ret):
        break

    # Resize and convert to rgb
    resized = cv2.resize(frame, (224,224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Preprocess (covert to tensor and normalize, then add bacth dimension)
    input = preprocess(rgb)
    input = torch.unsqueeze(input, 0)

    # Send to GPU
    input = input.to(device)

    # Inference
    output = custom_model(input)

    # Extract outputs
    output = output.cpu().detach().numpy()
    output = np.squeeze(output)
    resized = cv2.resize(output, (width,height))

    # Mask
    frame[:,:,2] = frame[:,:,0]/2 + (resized*255)
    frame[:,:,1] = frame[:,:,1]/2
    frame[:,:,0] = frame[:,:,2]/2

    # Display the resulting frame
    cv2.imshow('tracking', frame)

    # Save?
    if save:
        video.write(frame)

    # Wait for a keypress, and quit if 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the caputre
cap.release()
if save:
    video.release()

# Destroy display window
cv2.destroyAllWindows()

#FIN