# Intelligence : OpenCV
Running a pre-trained vision network can be quite easy using tools included with the newer versions of OpenCV.

## Setup
```bash
# Install Python packages
pip install opencv-python onnx
pip install numpy==1.26       # OpenCV is currently forcing numpy version >2 (which breaks other things)


# Create temporary folder for storing models and datasets
cd $LBB/boxes/intelligence/opencv
mkdir _tmp
cd _tmp
mkdir models
mkdir datasets
```

## Demos
- Face detection (bounding box and keypoints) using YuNet