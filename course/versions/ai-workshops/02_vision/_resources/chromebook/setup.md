## 1. Go to the workshop folder

```bash
cd /NoBlackBoxes/LastBlackBox/course/versions/ai-workshops/02_vision/_resources/chromebook
```

---

## 2. Create and activate the virtual environment

```bash
python3.10 -m venv vision_env
source vision_env/bin/activate
```

Check it:

```bash
which python
which pip
python --version
```

You want:

* `python` inside `.../vision_env/bin/python`
* `pip` inside `.../vision_env/bin/pip`
* Python `3.10.x`

---

## 3. Upgrade packaging tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## 4. Install NumPy first, but pin it to the compatible version

```bash
python -m pip install "numpy==1.26.4"
```
---

## 5. Install OpenCV from a binary wheel only

```bash
python -m pip install --only-binary=:all: "opencv-python==4.10.0.84"
```

---

## 6. Install Jupyter and ipykernel

```bash
python -m pip install jupyter ipykernel
```

---

## 7. Install PyTorch

```bash
python -m pip install torch torchvision torchaudio
```

---

## 8. Install Ultralytics

```bash
python -m pip install ultralytics
```

---

## 9. Install MediaPipe the safe way

First install its non-OpenCV dependencies manually:

```bash
python -m pip install "protobuf<5,>=4.25.3" flatbuffers sounddevice sentencepiece absl-py "jax==0.4.38" "jaxlib==0.4.38" "ml-dtypes==0.5.4"
```

Then install MediaPipe **without dependencies**:

```bash
python -m pip install --no-deps "mediapipe==0.10.21"
```

---

## 10. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name vision_env --display-name "Python (vision_env)"
```

---

## 11. Verify everything

Run these one by one:

```bash
python -c "import numpy as np; print('numpy ok', np.__version__)"
python -c "import cv2; print('cv2 ok', cv2.__version__)"
python -c "import torch; print('torch ok', torch.__version__)"
python -c "from ultralytics import YOLO; print('ultralytics ok')"
python -c "import mediapipe as mp; print('mediapipe ok', mp.__version__)"
python -c "import jupyter; import ipykernel; print('jupyter ok')"
```

---

## 12. Final notebook test

```bash
python
```

Then:

```python
import numpy as np
import cv2
import torch
import mediapipe as mp
from ultralytics import YOLO

print("Setup successful")
print("numpy", np.__version__)
print("cv2", cv2.__version__)
print("torch", torch.__version__)
print("mediapipe", mp.__version__)
```

---

## Full command block


```bash
cd /NoBlackBoxes/LastBlackBox/course/versions/ai-workshops/02_vision/_resources/chromebook

python3.10 -m venv vision_env
source vision_env/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install "numpy==1.26.4"
python -m pip install --only-binary=:all: "opencv-python==4.10.0.84"
python -m pip install jupyter ipykernel
python -m pip install torch torchvision torchaudio
python -m pip install ultralytics
python -m pip install "protobuf<5,>=4.25.3" flatbuffers sounddevice sentencepiece absl-py "jax==0.4.38" "jaxlib==0.4.38" "ml-dtypes==0.5.4"
python -m pip install --no-deps "mediapipe==0.10.21"
python -m ipykernel install --user --name vision_env --display-name "Python (vision_env)"
```