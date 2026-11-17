"""NB3 camera acquisition and SSD detection through the existing service."""
import sys
import logging
import threading
import time
from typing import NamedTuple

import cv2
import numpy as np

from config import REPO_ROOT, MODEL_PATH, LABELS_PATH, SOCKET_PATH


class Detection(NamedTuple):
    label: str
    score: float
    bbox: tuple  # normalized xmin, ymin, xmax, ymax


def load_labels(path):
    # Keep the unused 'n/a' slots: Coral uses 90 zero-based COCO category IDs.
    labels = path.read_text(encoding="utf-8").splitlines()
    if len(labels) != 90 or labels[0].strip() != "person" or labels[51].strip() != "banana":
        raise ValueError("Expected the matching 90-line, zero-based Coral COCO labels")
    return [label.strip() for label in labels]


def preprocess(frame, input_shape, input_dtype):
    """Convert a main-stream Picamera2 RGB888 (BGR byte order) frame to SSD RGB."""
    shape = tuple(input_shape)
    if len(shape) != 4 or shape[0] != 1 or shape[3] != 3 or input_dtype != "uint8":
        raise ValueError(f"Expected uint8 NHWC SSD input, got {shape}, {input_dtype}")
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a uint8, three-channel camera frame")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (shape[2], shape[1]), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized[np.newaxis, ...])


class Vision:
    def __init__(self, threshold):
        from rpi_edgetpu.client import EdgeTPUClient

        self.threshold = threshold
        self.labels = load_labels(LABELS_PATH)
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
        self.client = EdgeTPUClient(socket_path=SOCKET_PATH)
        try:
            self.metadata = self.client.load_model(str(MODEL_PATH))
            # Validate the model before opening the camera.
            preprocess(np.zeros((1, 1, 3), dtype=np.uint8),
                       self.metadata["input_shape"], self.metadata["input_dtype"])
        except BaseException:
            self.client.close()
            raise
        self.camera = None

    def start_camera(self, width, height):
        sys.path.insert(0, str(REPO_ROOT / "libs"))
        from NB3.Vision.camera import Camera

        self.camera = Camera(width=width, height=height)
        self.camera.start()

    def detect(self, frame):
        image = preprocess(frame, self.metadata["input_shape"], self.metadata["input_dtype"])
        results = self.client.detect(image, score_threshold=self.threshold)
        detections = []
        for result in results:
            class_id, score = result["class_id"], float(result["score"])
            if not 0 <= class_id < len(self.labels):
                raise ValueError(f"Unexpected COCO class ID: {class_id}")
            if self.labels[class_id] != "n/a" and np.isfinite(score) and score >= self.threshold:
                box = result["bbox"]
                coordinates = tuple(float(box[key]) for key in ("xmin", "ymin", "xmax", "ymax"))
                if not all(np.isfinite(coordinate) for coordinate in coordinates):
                    continue
                detections.append(Detection(self.labels[class_id], score, coordinates))
        return sorted(detections, key=lambda detection: detection[1], reverse=True)

    def close(self):
        try:
            if self.camera is not None:
                try:
                    self.camera.stop()
                finally:
                    self.camera.handle.close()
        finally:
            self.client.close()


class DetectorWorker:
    """One inference worker and one latest-JPEG mailbox; no camera access."""
    def __init__(self, threshold, on_result):
        self.condition = threading.Condition()
        self.pending = None
        self.running = True
        self.error = None
        self.threshold, self.on_result = threshold, on_result
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, jpeg, sequence, timestamp):
        with self.condition:
            self.pending = jpeg, sequence, timestamp
            self.condition.notify()

    def _run(self):
        vision = None
        try:
            vision = Vision(self.threshold)
            while self.running:
                with self.condition:
                    self.condition.wait_for(lambda: not self.running or self.pending is not None)
                    if not self.running:
                        break
                    jpeg, sequence, timestamp = self.pending
                    self.pending = None
                started = time.monotonic()
                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                detections = vision.detect(frame)
                self.on_result(detections, sequence, timestamp, 1000 * (time.monotonic() - started))
                with self.condition:
                    self.condition.wait_for(lambda: not self.running, timeout=max(0, .2 - (time.monotonic() - started)))
        except Exception as error:
            self.error = type(error).__name__
            from diagnostics import report_exception
            report_exception("Inference/response worker failed", error)
        finally:
            if vision is not None:
                vision.close()

    def close(self):
        with self.condition:
            self.running = False
            self.condition.notify_all()
        self.thread.join(timeout=35)
        if self.thread.is_alive():
            logging.getLogger("demo3.coral").warning("Inference worker did not stop within socket timeout")
