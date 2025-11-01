import os
import time
import cv2
import numpy as np
import NB3.Vision.camera as Camera         # NB3 Camera
#import NB3.Vision.webcam as Camera          # Webcam (PC)
import NB3.Server.server as Server

# Specify Paths
username = os.getlogin()
box_root = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/intelligence"
project_root = f"{box_root}/opencv"
site_root = f"{project_root}/sites/single"
model_path = f"{project_root}/_tmp/models/face_detection_yunet_2023mar.onnx"

# Setup Camera
camera = Camera.Camera(width=640, height=480, lores_width=640, lores_height=480)
camera.overlay.timestamp = True
camera.start()

# Start Server (for streaming)
interface = Server.get_wifi_interface()
server = Server.Server(root=site_root, interface=interface, autostart=True)

# Setup Detector
score_threshold = 0.9
nms_threshold = 0.3
top_k = 5000
detector = cv2.FaceDetectorYN_create(
    model=model_path,
    config="",
    input_size=(640, 480),
    score_threshold=score_threshold,
    nms_threshold=nms_threshold,
    top_k=top_k,
    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
    target_id=cv2.dnn.DNN_TARGET_CPU
)

# Detected Face Processing Function
def process_faces(detections):
    results = []
    if detections is None:
        return results
    for detection in detections:
        detection = detection.astype(np.float32)
        x, y, w, h = detection[:4]
        kps = detection[4:14].reshape(5, 2)
        score = float(detection[-1])
        results.append({
            "bbox": (int(x), int(y), int(w), int(h)),
            "kps": [(int(px), int(py)) for px, py in kps],
            "score": score
        })
    return results

# Processing Loop
try:
    while True:
        # Capture frame
        frame = camera.capture(mjpeg=False, lores=False, gray=False)

        # Detect and process faces
        detections = detector.detect(frame)[1]
        faces = process_faces(detections)

        # Overlay detections (rectangles and keypoints)
        camera.overlay.clear()
        for face in faces:
            x, y, w, h = face["bbox"]
            camera.overlay.add_rectangle(x, y, w, h)
            for kp in face["kps"]:
                x, y = kp
                camera.overlay.add_circle(x, y, 2)

        # Display raw frame with detections overlaid
        camera.display(frame, server, "display", overlay=True, jpeg=False, gray=False)
        
        # Delay
        time.sleep(0.033) # Limit to 30 FPS

except KeyboardInterrupt:
    server.stop()
    camera.stop()
