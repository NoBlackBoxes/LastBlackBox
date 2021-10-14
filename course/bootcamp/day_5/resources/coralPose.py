### Testing the Edge TPU
import cv2
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

_NUM_KEYPOINTS = 17

# Open an image using OpenCV, change color space for Deep Neural Network
img = cv2.imread('test.jpg')

# This is where you specify the Deep Neural Network.
# Please put it in the same folder as the python file.
# --> this can go at the very beginning after import cv2 in the streaming file
interpreter = make_interpreter('movenet_single_pose_lightning_ptq_edgetpu.tflite')
interpreter.allocate_tensors()
## Until here

# Convert to RGB, not needed for streaming
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#### --> needs to happen for each image ####
# This resizes the RGB image
resized_img = cv2.resize(img_rgb, common.input_size(interpreter))
# Send resized image to Coral
common.set_input(interpreter, resized_img)

# Do the job
interpreter.invoke()

# Get the pose
pose = common.output_tensor(interpreter, 0).copy().reshape(_NUM_KEYPOINTS, 3)

height, width, ch = img.shape

# Draw the pose onto the image using blue dots
for i in range(0, _NUM_KEYPOINTS):
    cv2.circle(img,
               [int(pose[i][1] * width), int(pose[i][0] * height)],
               5, # radius
               (255, 0, 0), # color in RGB
               -1) # fill the circle
    
cv2.imwrite('test_pose.jpg', img)