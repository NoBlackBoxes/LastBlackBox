"""Demo settings; paths are independent of the current working directory."""
from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[6]
AUDIO_PYTHON = REPO_ROOT / "course/versions/ai-workshops/01_audio/_resources/python"
# MODEL_PATH = REPO_ROOT / "_tmp/coral/models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
MODEL_PATH = REPO_ROOT / "_tmp/coral/models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite"
LABELS_PATH = MODEL_PATH.parent / "coco_labels.txt"
SOCKET_PATH = "/tmp/edgetpu.sock"
ALSA_DEVICE = os.environ.get("NB3_ALSA_DEVICE", "plughw:CARD=MAX98357A,DEV=0")
CONFIDENCE = 0.65
STABLE_FRAMES = 3
COOLDOWN_SECONDS = 8.0
FRAME_INTERVAL = 0.2
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480
