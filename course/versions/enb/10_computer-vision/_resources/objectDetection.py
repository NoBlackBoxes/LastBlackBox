import cv2
import torch
import torchvision
import torchvision.transforms as T

# Load pretrained SSDlite model (lightweight for RPi4)
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
model.eval()

# COCO dataset labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Transform: resize to 320x320 and convert to tensor
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((320, 320)),
    T.ToTensor()
])

def detect_and_draw(frame, model, threshold=0.5):
    H, W = frame.shape[:2]   # original size

    # Resize and convert to tensor
    img_resized = transform(frame)
    with torch.no_grad():
        preds = model([img_resized])[0]

    boxes = preds['boxes']
    labels = preds['labels']
    scores = preds['scores']

    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            # Scale box coords back to original resolution
            x1, y1, x2, y2 = box.tolist()
            x1 = int(x1 / 320 * W)
            x2 = int(x2 / 320 * W)
            y1 = int(y1 / 320 * H)
            y2 = int(y2 / 320 * H)

            class_name = COCO_INSTANCE_CATEGORY_NAMES[label]

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {score:.2f}",
                        (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame
