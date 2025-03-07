# The Last Black Box : Vision
In this box, you will learn about vision...

## Vision
Look at everyone.

<details><summary><i>Materials</i></summary><p>

Name|Depth|Description| # |Data|Link|
:-------|:---:|:----------|:-:|:--:|:--:|
Camera (RPi v3)|01|RPi color camera with auto-focus (version 3)|1|[-D-](/boxes/vision/_resources/datasheets/rpi_camera_v3.pdf)|[-L-](https://uk.farnell.com/raspberry-pi/sc0872/rpi-camera-mod-3-standard-lens/dp/4132318)
NB3 Camera Mount|01|Custom laser cut mount for RPi camera|1|[-D-](/boxes/vision/NB3_camera_mount)|[-L-](VK)
NB3 Cortex Mount|01|Custom laser cut holder for NPU|1|[-D-](/boxes/vision/NB3_cortex_mount)|[-L-](VK)
M2.5 bolt (6)|01|6 mm long M2.5 bolt|4|[-D-](/boxes/robotics/)|[-L-](https://www.accu.co.uk/pozi-pan-head-screws/9255-SPP-M2-5-6-A2)
M2.5 standoff (20/PS)|01|20 mm long plug-to-socket M2.5 standoff|4|[-D-](/boxes/vision/)|[-L-](https://uk.farnell.com/wurth-elektronik/971200151/standoff-hex-male-female-20mm/dp/2884418)
M3 nut (square)|01|square M3 nut 1.8 mm thick|1|[-D-](/boxes/audio/-)|[-L-](https://www.accu.co.uk/flat-square-nuts/21326-HFSN-M3-A2)
M3 bolt (12)|01|12 mm long M3 bolt|1|[-D-](/boxes/vision/)|[-L-](https://www.accu.co.uk/pozi-pan-head-screws/500116-SPP-M3-12-ST-BZP)
M2 bolt (8)|01|8 mm long M2 bolt|4|[-D-](/boxes/audio/)|[-L-](https://www.accu.co.uk/pozi-pan-head-screws/500101-SPP-M2-8-ST-BZP)
M2 nut|01|regular M2 nut|4|[-D-](/boxes/audio/)|[-L-](https://www.accu.co.uk/hexagon-nuts/456429-HPN-M2-C8-Z)

</p></details><hr>

### Cameras
> From CCDs to CMOS.


### Image-Processing
> Old-school (classical) techniques for processing two-dimensional data (2D).

- Install python opencv
```bash
pip install opencv-python
```


# Project
### NB3 : Streaming Images
> Let's stream live images from your NB3's camera to any web browser.


### NB3 : Install the Eye
> Let's install a camera on your NB3.

<details><summary><weak>Guide</weak></summary>
:-:-: A video guide to completing this project can be viewed <a href="https://vimeo.com/1042945461" target="_blank" rel="noopener noreferrer">here</a>.
</details><hr>

- Test your camera with the following command
```bash
# Navigate to LBB root
cd ~/NoBlackBoxes/LastBlackBox


# Create folder for images in the temporary folder
mkdir _tmp/images


# Navigate to the temporary image folder
cd _tmp/images


# Capture a still image and save it to a file
libcamera-still -o test.png
```


