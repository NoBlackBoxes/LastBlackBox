# Vision : Image Processing : picamera

This directory contains examples using the **picamera2** library, which provides Python bindings for the Raspberry Pi's libcamera stack.

## About picamera2

picamera2 is the successor to the original picamera library and uses the modern libcamera framework. It provides:
- Direct access to Raspberry Pi camera hardware
- Support for multiple camera configurations (preview, still, video)
- Hardware-accelerated encoding (MJPEG, H.264)
- Transform operations (flip, rotate)

## Installation

The picamera2 library is pre-installed on Raspberry Pi OS (Bullseye and later). For manual installation:

```bash
sudo apt update
sudo apt install python3-picamera2
```

**Note**: picamera2 only works on Raspberry Pi hardware. For development on PC, the Python scripts in `../python/` automatically fall back to webcam support.

## Example Scripts

1. **`00_still.py`** - Captures a single still image as PNG
2. **`01_still_flip.py`** - Captures a still image with vertical flip transform applied

## Usage

**⚠️ Setup Required**: Before running the scripts, make sure you have set up the LBB Python virtual environment. Follow the [Virtual Environment Setup Guide](../../../python/virtual_environments/README.md) first.

Navigate to the directory first, then run the scripts:
```bash
cd ~/NoBlackBoxes/LastBlackBox/boxes/vision/image_processing/picamera
python 00_still.py
python 01_still_flip.py
```

Images are saved to the current directory as `my_still_flip.png`.

## Integration with Python Scripts

The Python scripts in `../python/` use the `NB3.Vision.camera` module, which wraps picamera2 functionality and provides a unified interface that works on both Raspberry Pi (picamera2) and PC (webcam).
