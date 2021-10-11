Machine Vision
==============

Afternoon session

You need
--------

-   The working cameraStream.py from the morning session

-   Your computer and SSH connection to Raspberry Pi 4

Goal
----

You are able to use basic image processing, not only locally on your
machine, but also on the live stream, and thus, being able to
dynamically interact with the robot.


Task 1: Incorporate the face detection algorithm
------------------------------------------------

I prepared a little script called “faceDetection.py” that contains all
the code of the face detection algorithm. Ensure you understand what
each line of code is doing. Run it locally on your computer to see if it
works in general. Your aim is it to incorporate this code into the
cameraStream.py file to allow live face recognition with your robot.

Tipp: 
- Check the variable names
- Check the detector parameters

![](./media/image9.jpeg)

Task 2: Save face images
------------------------

Collect at least **100 images** from your face. One approach is to save the
complete images, but you can also only save the crops to your face.

Tipps: 
- How do you crop an image using numpy array indexing?
- How do you save images?
- Is there a way you could think of to give them a names like 0.jpg,
1.jpg, … ?

Task 3: Let the robot react
---------------------------

If you detect a face using your face detection algorithm, let the robot
interact with the environment. For example, let an LED light up or make
some noise using the speaker.

Task 4: Resizing images
-----------------------

Your face images should have at the very end all the same size, just
like (128, 128, 3). Find a function to resize all images (one by one)
and save them accordingly. Please create a video using imageio similar
to the one I’ve shown you in the lecture.

Result
------

You should have a gallery of (128, 128, 3) images of your face and your
live stream detects your face in real-time.

![](./media/image10.jpeg)

Pro-Task 1: Incorporate background subtraction to your live stream
------------------------------------------------------------------

Use a temporary variable that stores the information of the background.
You have to define it below the class, similar to this one. Think about
it, what does this mean?

![](./media/image11.png)

You can use the bg variable in your image processing part by referring
to it as `self.bg`.

If you want to replace image 0, you can write `self.bg\[0\] = …`

Ensure that you are cycling through your background images and always
keep the five latest background images! Then subtract the average
background from your camera frame. Be aware of the live lecture!

![](./media/image12.png)

Pro-Task 2: Threshold the background substracted frame
------------------------------------------------------

Use thresholding to generate a black/white image. Problems you may run
into: OpenCV wants uint8 images. Is there an OpenCV Thresholding
function? Which threshold is good?

Pro-Task 3: Find the center of mass of your thresholded image
-------------------------------------------------------------

You can use image moments (is there an OpenCV function…?) to find the
centroid x and y coordinates. Checkout appropriate tutorials how to
compute these locations. Then, draw an ellipse or circle with a given
radius there. Maybe try to use color?
