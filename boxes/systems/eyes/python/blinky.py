import NB3.Eyes.eye as Eye
import time

right_eye = Eye.Eye(0)
left_eye = Eye.Eye(1)

right_eye.set_all(True)
left_eye.set_all(True)

for i in range(0,16,1):
    right_eye.set_intensity(i)
    left_eye.set_intensity(i)
    time.sleep(0.01)
for i in range(15,-1,-1):
    right_eye.set_intensity(i)
    left_eye.set_intensity(i)
    time.sleep(0.01)

right_eye.set_all(False)
left_eye.set_all(False)

right_eye.close()
left_eye.close()

