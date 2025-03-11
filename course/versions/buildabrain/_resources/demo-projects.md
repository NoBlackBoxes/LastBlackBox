# NB3 Demo Projects
These example projects demonstrate what the NB3 can do *and* provide a "starting point" for developing your own ideas.

## Drone
A first-person-view (FPV) "Drone" controller for your NB3. Sends a live-feed from the camera to a website, which you can open on your PC or Phone. The website provides a simple interface to control the robot's motion...but could control much, much more!

```bash
# Drone demo shortcut
Drone
# Follow the link to view the web interface
```
The Python code for "Drone" demo is [here](/boxes/vision/drone-NB3/).

## Listen
Uses the NB3's ears to continuously record audio and a small neural network (running on the NPU) to detect "voice commands". Currently, this demo only responds to the command "turn left" by...turning left. However, the list of possible commands is [here](/boxes/intelligence/NPU/listen-NB3/model/labels.txt). How would you like your NB3 to respond?

```bash
# Listen demo shortcut
Listen
```
The Python code for "Listen" demo is [here](/boxes/intelligence/NPU/listen-NB3/).

## Look
Uses the NB3's eye to continuously record video and a convolutional neural network (running on the NPU) to detect "faces". This demo only detects one face and draws a rectangle around it, streaming the results to a website for remote viewing. However, the NB3 could respond in many other ways detecting a face. It could follow it? Say hello?

```bash
# Look demo shortcut
Look
# Follow the link to view the image stream
```
The Python code for "Look" demo is [here](/boxes/intelligence/NPU/look-NB3/).

## Other Demos
The full list of demo "shortcuts" (aliases) are listed here for your convenience:

```bash
Sync    # Sync with the latest LBB repo (and these bash aliases)
Drive   # Run the python driving controller (for remotely-controlling NB3)
Stream  # Run the camera streaming (camera images will be streamed to the indicated website)
Drone   # Run a web interface to both view the camera and drive the NB3
Listen  # Run the keyword detection model and respond to commands
Look    # Run the face tracking model and follow detected faces
Music   # Play some music through the speaker
```