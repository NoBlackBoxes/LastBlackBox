Machine Vision
==============

Morning session

You need
--------

-   NB3 Robot

-   Python Camera module

-   WiFi/SSH access to Raspberry Pi

Goal
----

Today’s task is it to provide another sense to your robot: vision! We
therefore will attach a camera to the Raspberry Pi 4 and stream the data
via a webserver to your computer and/or smartphone. Finally, we
implement some basic image processing operations to enable cortical
vision processing, such as thresholding and object recognition.

Task 1: Enabling the camera
---------------------------

First, please attach the camera to your Raspberry Pi 4. A detailed video
how to attach the camera with the flat ribbon cable to the Raspberry Pi
4 is provided here:

<https://www.raspberrypi.com/documentation/accessories/camera.html>

If you attached the camera, you need to place the camera on the provided
camera holders in order to mount everything on the robot (see also Video
from Adam regarding camera installation and mounting). Next, you need to
enable the camera in your Raspberry Pi 4 with SSH.

Remember, you can SSH on your raspberry pi using PuTTY or KiTTY using
your set username/password (by default pi and raspberry, respectively).
Next, use the raspi-config (`sudo raspi-config`) to enable the camera:

![](./media/image1.png)

Task 2: Take a first photo and video
------------------------------------

Using SSH, you can interact with the camera using raspistill and
raspivid. A full explanation about the settings of these two commands is
here:

<https://blog.robertelder.org/commands-raspberry-pi-camera/>

The files are saved in the same folder as your currently in (e.g. your
home folder), but you can specify the path.

For taking a single photo you could for example use:

    raspistill -w 640 -h 480 -q 75 -vf -o test_image.jpg
    (image with 640x480 pixel, vertically flipped, saved as test_image.jpg with a quality of 75 (lower worse))

    raspivid -o test_vid.h264 -t 5000 -fps 24
    (video with default 1920x1080 px (full HD), saved as test_vid.h264 and
    records at 24 frames per second for 5 s (5000 ms))

This records RAW h264 footage, and is not contained in a container file.
The most common container file is the MP4 format. If the raw footage is
not in a container, VLC or other players will refuse to play it,
however, you can still open these videos via Python and e.g. imageio.
Using *MP4Box* or FFMPEG you can easily add an MP4 container, though.

First, install MP4Box if it is not pre-installed:

    sudo apt-get install gpac

Next, use the following command to convert the raw video file to an mp4
container.

    MP4Box -add test_vid.h264 test_vid.mp4

Task 3: Transfer them to your computer
--------------------------------------

You can use for example FileZilla
([www.filezilla-project.org](https://filezilla-project.org/download.php?show_all=1)) to transfer files
easily.

Setup your connection the following way by adding a new “Server”:

![](./media/image2.png)


I know it is in German, but you just click “New Server”, use the
protocol SFTP, add the IP address of your Raspberry Pi (In the
screenshot is MY IP-Address, which is not necessarily yours!), set the **port to 22** and add
the username/password for connection. Next, click on “Connect”, and you
will see something like this (the port is missing though):

![](./media/image3.png)

On the left is your PC, on the right is your Rapsberry Pi. By using
drag&drop, you can move files from your PC to the Raspberry Pi and vice
versa. For example, you can now navigate to the folder, where you saved
your videos (e.g. `/home/pi/…`) and transfer them to your computer. Try to
open the files and watch the video (check both, h264 – that shouldn’t
work, and mp4 – that should work).

Task 4: Live stream the camera feed
-----------------------------------

I prepared a file called cameraStream.py – this is based on the basic
streaming example of **picamera**
(<https://picamera.readthedocs.io/en/release-1.13/recipes2.html#web-streaming>)
and a little bit adjusted for the afternoon tasks. It consists of an
http server allowing to stream dynamically an image that changes over
time (the file stream.mjpg). I further added some libraries for image
processing, namely numpy[^1] and OpenCV[^2]. It is necessary to install
both libraries. Here’s how you do it:

    pip3 install opencv-contrib-python

    sudo apt-get install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev
    libatlas-base-dev libjasper-dev libqtgui4 libqt4-test

    pip3 install numpy --upgrade

If you have any issues here, let us know. 
Also, if you are using the **headless Lite OS**, it may happen that you need to install a couple of more libraries: 

    sudo apt-get python3-opencv
    sudo apt-get python3-pip

Next, transfer the `cameraStream.py` file somewhere on your Raspberry Pi, e.g. in
`/home/pi/Documents`. You can then execute it using the following command (starting with `python`):

    \~$ python3 Documents/cameraStream.py

On your computer or cell phone that is in the same network as your
Raspberry Pi, go to an internet browser (Firefox, Chrome, …), type in
the Raspberry Pi’s IP address with the port 8000, e.g.

    http://192.168.43.78:8000

and now you should see a live stream of your Raspberry Pi’s camera

Task 5: Transfer and open a photo
---------------------------------

The last task for the morning is to transfer a photo from your Raspberry
Pi to your computer. Use either Jupyter notebooks or a Python script
inside of Visual Studio Code to visualize the image (see paragraphs below). Try to work with
the image, for example crop it, or resize it, etc. This knowledge is
important for the afternoon and tomorrow.

I am assuming that you have either Anaconda installed on your machine (PC!) or 
any other Python version that has Juptyter installed.

Results
-------

This is how your robot should look like:

![](./media/image3a.png)

And this is your camera stream

![](./media/image4.png)

cameraStream.py
---------------

Here’s a quick overview of the file `cameraStream.py`:

First, we important libraries important for camera interaction and
live-streaming:

![Ein Bild, das Text enthält. Automatisch generierte
Beschreibung](./media/image5.png)

Next, we import libraries for dealing with the images:

![](./media/image6.png)

(These are the libraries you just installed)

Then, there is a lot of stuff related to the webserver, the real
important things are going on here:

![](./media/image7.png)

Between both decoding and encoding steps, you are able to do anything to
the image ( afternoon tasks), here introduced as variable frame.

Open, Modify and Save Images
----------------------------

Not so much Python experience? Watch this video (ca. first 13 minutes):

<https://www.youtube.com/watch?v=Z1Yd7upQsXY>

Never heard of numpy? Check out the numpy intro:
<https://numpy.org/doc/stable/user/quickstart.html>

Images can be interpreted as numpy ndarrays of a given shape:

    (1080, 1920, 3) e.g. Full HD (1920 px wide, 1080 px high, 3 color
    channels)

    (20, 512, 512) e.g. z-stack of 20 planes, of 512x512 px images

    (5, 30, 256, 256, 3) e.g. time-lapse (5 ts) z-stack (30 planes) of
    256x256 px images

having 3 channels (e.g. GFP, mCherry, brightfield)

Use jupyter notebooks (preferred for this course) or python scripts to
open, modify and save images. Use the imageio library to open and save
images.

    im = io.imread(“test_image.jpg”)

    io.imwrite(“cropped.jpg”, im[10:, 30:])

    # skips 30 px in x and 10 px in y

Alternatively, you can use OpenCV to load and handle images. This is
also recommended for your Raspberry Pi. Commands that come in handy are:

    im = cv2.imread(“test_image.jpg”)

    cv2.imwrite(“cropped.jpg”, im[10:, 30:])

    cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # images loaded with OpenCV have the color channels ordered
    # differently, i.e. Blue, Green, Red (BGR)

I recommend you the introduction of scikit-image explaining how to work
with images and how to use numpy for image manipulation:
<https://scikit-image.org/docs/dev/user_guide.html>, especially the
chapter “[A crash course on NumPy for
images](https://scikit-image.org/docs/dev/user_guide/numpy_images.html)”.

How to open a Jupyter notebook after installing Anaconda:

![](./media/image8.png)

[^1]: Numpy is a nice package for dealing with mathematics and arrays.
    <https://numpy.org/> and look at their Nature paper:
    <https://www.nature.com/articles/s41586-020-2649-2>

[^2]: OpenCV is a great open library for computer vision written in C++
    with bindings for Python. <https://opencv.org/> Also checkout
    scikit-image, another powerful library for Python.
