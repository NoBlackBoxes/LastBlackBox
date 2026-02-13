# Vision : Image Processing : Python

This directory contains Python scripts for image processing, organized into two categories: **offline** (static image processing) and **online** (live video stream processing with web server integration).

## Structure

- **`offline/`** - Scripts that process static images (single frames)
- **`online/`** - Scripts that process live video streams with web server integration
- **`online/sites/`** - Web interfaces for streaming visualization

## Offline Scripts

These scripts process saved images:

1. **`00_still.py`** - Captures a single frame from camera (NB3 or webcam) and saves it
2. **`01_threshold.py`** - Applies binary thresholding to a grayscale image
3. **`02_edges.py`** - Canny edge detection on thresholded images
4. **`03_rgb_vs_hsv.py`** - Visualizes RGB vs HSV color spaces with matplotlib
5. **`04_hsv_threshold.py`** - HSV color thresholding to isolate red color (handles circular hue)
6. **`05_find_largest_blob.py`** - Finds the largest blob in HSV-thresholded images and draws bounding boxes

## Online Scripts

These scripts process live video streams:

1. **`01_threshold.py`** - Real-time binary thresholding with split-screen web display
2. **`01i_threshold.py`** - Interactive thresholding with web slider to adjust threshold level
3. **`02_frame-by-frame.py`** - Motion detection via frame differencing (absolute difference)
4. **`03_hsv_centroid.py`** - HSV color tracking for green objects, finds largest blob and displays centroid
5. **`03i_hsv_centroid.py`** - Interactive HSV tracking with sliders for H/S/V thresholds
6. **`04_faces.py`** - Face detection using Viola-Jones Haar Cascade classifier

## Key Features

- **Camera Support**: Works with NB3 (Raspberry Pi camera) or PC webcam
- **Web Streaming**: Uses `NB3.Server.server` to stream processed video to web browsers
- **Multiple Display Modes**: Split-screen, single view, and interactive slider interfaces
- **Overlay Annotations**: Timestamps, bounding boxes, labels, and coordinates
- **Color Spaces**: RGB, grayscale, and HSV processing
- **Computer Vision**: Thresholding, edge detection, color tracking, blob detection, and face detection

## Usage

### Offline Scripts

Run scripts sequentially (they build on each other):
```bash
python offline/00_still.py    # Capture image
python offline/01_threshold.py # Process image
# etc.
```

### Online Scripts

Start a script and access the web interface:
```bash
python online/01_threshold.py
# Then open browser to the displayed URL
```

## Dependencies

- `opencv-python` (cv2)
- `numpy`
- `matplotlib` (for offline scripts)
- `NB3.Vision.camera` or `NB3.Vision.webcam`
- `NB3.Server.server`
- `LBB.config`
