# Vision Workshop Setup (Chromebook / Linux)

## 1. Go to the workshop folder

``` bash
cd ~/NoBlackBoxes/LastBlackBox/course/versions/ai-workshops/02_vision/_resources/chromebook
```

------------------------------------------------------------------------

## 2. Check Python and install venv support

``` bash
python3 --version
sudo apt update
sudo apt install python3.11-venv
```

------------------------------------------------------------------------

## 3. Create and activate the virtual environment

``` bash
python3 -m venv vision_env
source vision_env/bin/activate
```

Check it:

``` bash
which python
which pip
python --version
```

Expected: - Python path inside `vision_env` - Pip path inside
`vision_env` - Python 3.11.x

------------------------------------------------------------------------

## 4. Upgrade packaging tools

``` bash
python -m pip install --upgrade pip setuptools wheel
```

------------------------------------------------------------------------

## 5. Install NumPy

``` bash
python -m pip install "numpy==1.26.4"
```

------------------------------------------------------------------------

## 6. Install OpenCV

``` bash
python -m pip install --only-binary=:all: "opencv-python==4.10.0.84"
```

------------------------------------------------------------------------

## 7. Install Jupyter

``` bash
python -m pip install jupyter ipykernel
```

------------------------------------------------------------------------

## 8. Install PyTorch (CPU ONLY)

⚠️ Important: prevents huge CUDA downloads

``` bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

------------------------------------------------------------------------

## 9. Install Ultralytics

``` bash
python -m pip install ultralytics
```

------------------------------------------------------------------------

## 10. Install MediaPipe

``` bash
python -m pip install "protobuf<5,>=4.25.3" flatbuffers sounddevice sentencepiece absl-py "jax==0.4.38" "jaxlib==0.4.38" "ml-dtypes==0.5.4"
python -m pip install --no-deps "mediapipe==0.10.21"
```

------------------------------------------------------------------------

## 11. Register Jupyter kernel

``` bash
python -m ipykernel install --user --name vision_env --display-name "Python (vision_env)"
```

------------------------------------------------------------------------

## 12. Verify installation

``` bash
python -c "import numpy as np; print('numpy ok', np.__version__)"
python -c "import cv2; print('cv2 ok', cv2.__version__)"
python -c "import torch; print('torch ok', torch.__version__)"
python -c "from ultralytics import YOLO; print('ultralytics ok')"
python -c "import mediapipe as mp; print('mediapipe ok', mp.__version__)"
python -c "import jupyter; import ipykernel; print('jupyter ok')"
```

------------------------------------------------------------------------

## 13. VS Code Notes

-   Open the `chromebook` folder only
-   Select interpreter:

``` text
.../chromebook/vision_env/bin/python
```

-   Select kernel:

``` text
Python (vision_env)
```

------------------------------------------------------------------------

## Full Setup (copy-paste)

``` bash
cd ~/NoBlackBoxes/LastBlackBox/course/versions/ai-workshops/02_vision/_resources/chromebook

python3 --version
sudo apt update
sudo apt install python3.11-venv

python3 -m venv vision_env
source vision_env/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install "numpy==1.26.4"
python -m pip install --only-binary=:all: "opencv-python==4.10.0.84"
python -m pip install jupyter ipykernel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install ultralytics
python -m pip install "protobuf<5,>=4.25.3" flatbuffers sounddevice sentencepiece absl-py "jax==0.4.38" "jaxlib==0.4.38" "ml-dtypes==0.5.4"
python -m pip install --no-deps "mediapipe==0.10.21"
python -m ipykernel install --user --name vision_env --display-name "Python (vision_env)"
```
