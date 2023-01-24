import cv2
import torch
import model
from torchvision import transforms

# Specify video or camera
live_capture = True
if live_capture:
    width = 640
    height = 480
else:
    width = 512
    height = 512

# Specify paths
repo_path = '/home/kampff/NoBlackBoxes/repos/OtherBlackBoxes'
box_path = repo_path + '/boxes/ai/tracking/nose/regression'
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

# Loop until 'q' pressed
smooth = True
alpha = 0.5
first_frame = True
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
    print(output)
    x = output[0,0] * width
    y = output[0,1] * height

    # Smooth?
    if smooth and (not first_frame):
        _x = (alpha * x) + ((1 - alpha) * _x)
        _y = (alpha * y) + ((1 - alpha) * _y)
    else:
        _x = x
        _y = y
    first_frame = False

    # Draw tracked point
    frame = cv2.circle(frame, (int(_x), int(_y)), 11, (0, 255, 255), thickness=-1)

    # Display the resulting frame
    cv2.imshow('tracking', frame)

    # Wait for a keypress, and quit if 'q'
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Release the caputre
cap.release()

# Destroy display window
cv2.destroyAllWindows()

#FIN