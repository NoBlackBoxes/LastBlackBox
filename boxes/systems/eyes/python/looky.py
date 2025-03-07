import NB3.Eyes.eye as Eye
import time

right_eye = Eye.Eye(0)
left_eye = Eye.Eye(1)
right_eye.set_intensity(0)
left_eye.set_intensity(0)

# Set gaze direction
right_eye.saccade(4,4,300)
left_eye.saccade(4,4,300)
time.sleep(0.5)
right_eye.saccade(6,6,300)
left_eye.saccade(6,6,300)
time.sleep(1.0)
right_eye.saccade(3,2,320)
left_eye.saccade(3,2,320)
time.sleep(0.75)
right_eye.saccade(2,4,300)
left_eye.saccade(2,4,300)
time.sleep(0.5)

# Clear
right_eye.clear()
left_eye.clear()

# Close
right_eye.close()
left_eye.close()

