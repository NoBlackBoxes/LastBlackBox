# Vision Workshop --- Setup

This guide explains how to set up the Python environment required for
the vision workshop notebooks.

------------------------------------------------------------------------

## Requirements

-   Python **≥ 3.9**
-   Git installed

⚠️ **Important (Anaconda users)**\
If your terminal prompt shows `(base)`, you are inside a Conda
environment.\
Deactivate it before continuing:

``` bash
conda deactivate
```

The workshop environment should be created **outside Conda**.

------------------------------------------------------------------------

## 1. Clone the repository

``` bash
git clone https://github.com/NoBlackBoxes/LastBlackBox.git
cd LastBlackBox/course/versions/ai-workshops/02_vision
```

This directory should contain:

    requirements.txt
    notebooks/

------------------------------------------------------------------------

## 2. Create a virtual environment

Create a dedicated Python environment for the workshop.

``` bash
python3 -m venv vision_env
```

Activate it.

### Mac / Linux

``` bash
source vision_env/bin/activate
```

### Windows

``` bash
vision_env\Scripts\activate
```

Your terminal should now show:

    (vision_env)

------------------------------------------------------------------------

## 3. Verify the environment

Check that Python and pip come from the virtual environment:

``` bash
which python
which pip
```

Both paths should contain:

    vision_env

and **not** `anaconda3`.

------------------------------------------------------------------------

## 4. Install dependencies

Upgrade packaging tools and install dependencies.

``` bash
pip install --upgrade pip setuptools wheel
pip install --only-binary=:all: opencv-python
pip install -r requirements.txt
pip install ipykernel
```

This installs the libraries used during the workshop, including:

-   NumPy
-   OpenCV
-   PyTorch
-   Ultralytics (YOLO)
-   Jupyter

------------------------------------------------------------------------

## 5. Register the environment in Jupyter

Register the environment so it appears as a selectable kernel.

``` bash
python -m ipykernel install --user --name vision_env
```

------------------------------------------------------------------------

## 6. Start Python

Launch python:

``` bash
python
```

------------------------------------------------------------------------

## 7. Verify the installation

Run the following cell in a notebook:

``` python
import numpy as np
import cv2
import torch
from ultralytics import YOLO

print("Setup successful")
```

If no errors appear, the environment is ready.

------------------------------------------------------------------------

## Ready to start

Open the notebooks in:

    notebooks/

and begin the vision workshop.
