import time
import io
import threading
import cv2

class Camera(object):
    thread = None  # background thread that reads frames from camera
    frame = None  # current frame is stored here by background thread
    last_access = 0  # time of last client access to the camera

    def initialize(self):
        if Camera.thread is None:
            # start background frame thread
            Camera.thread = threading.Thread(target=self._thread)
            Camera.thread.start()

            # wait until frames start to be available
            while self.frame is None:
                time.sleep(0)

    def get_frame(self):
        Camera.last_access = time.time()
        self.initialize()
        return self.frame

    @classmethod
    def _thread(cls):
        
        camera = cv2.VideoCapture(0)
        stream = io.BytesIO()
        while(True):
            # Read most recent frame
            ret, frame = camera.read()

            # Encode as JPEG
            is_success, im_buf_arr = cv2.imencode(".jpg", frame)
            cls.frame = im_buf_arr.tobytes()

            # if there hasn't been any clients asking for frames in
            # the last 10 seconds stop the thread
            if time.time() - cls.last_access > 10:
                break
                camera.release()
        cls.thread = None
