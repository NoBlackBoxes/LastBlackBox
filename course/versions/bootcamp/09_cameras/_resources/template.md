# Bootcamp : Cameras

## Vision

### Watch this video: [Cameras](https://vimeo.com/??????)
> How does a digital camera work?

### Watch this video: [NB3 Forebrain](https://vimeo.com/628545232)
> Let's add a camera (eyes) to your NB3!

- [] **TASK**: Check whether your camera works by running the following command from the Linux terminal.
```bash
libcamera-still -o test.png
```
- *Hint*: This command will take a picture and save it as a PNG in the same folder where you run the command. You can open it directly in VS Code.


## Computer Vision

Taking a picture is just the first step in making a computer "see". How do we extract useful information from the image? How do we detect movement? How do we identify and localise different objects? Many of these tasks are surprisingly hard and remain unsolved problems in both engineering and neuroscience?

Let's get started.

The first step will be acquiring an image using Python (so we can then manipulate and analyse the image using code).

### Acquire an image using Python

- Run this [Example](resources/python/camera/capture_save.py) code to acquire a still image and save it to a file.

### Process an image using Python and the Numpy library

- Run this [Example](resources/python/camera/capture_process_save.py) code to acquire a still image and save it to a file.
- Note: you may need to install another library to save the numpy array as an image

  ```bash
  pip install pillow
  ```

### Stream images from your NB3 camera to the local network (so we can view the live stream on your laptop)

- Run this [Example](resources/python/camera/capture_stream.py) code to continously acquire images and stream them to a website.

----

## Afternoon

Project Work

----



  ----