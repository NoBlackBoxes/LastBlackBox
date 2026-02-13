# Vision : Image Processing

This directory contains image processing examples and tools for computer vision tasks.

## Subdirectories

- **`python/`** - Python scripts for image processing (offline and online/streaming)
- **`picamera/`** - Information about using picamera2 library with libcamera

## Quick Start

### Python Dependencies

```bash
pip install opencv-python
pip install numpy==1.26       # OpenCV is currently forcing numpy version >2 (which breaks other things)
```

### Camera Support

- **NB3/Raspberry Pi**: Uses picamera2 library (pre-installed on Raspberry Pi OS)
- **PC/Development**: Falls back to webcam support

See the individual subdirectory READMEs for detailed documentation:
- [`python/README.md`](python/README.md) - Python scripts documentation
- [`picamera/README.md`](picamera/README.md) - picamera2 information
