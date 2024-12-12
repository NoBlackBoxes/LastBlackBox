# Python : Virtual Environments

Virtual environments are self-contained installations of Python. All of the packages you install and changes you make only affect this *local* environment.

---
## Create a virtual environment

- Make a sub-folder in the **repository root** called "_tmp"
  - *Note*: Anything in the "_tmp" folder is ignored by Git and not synced to the main repository
- Create a Python virtual environment (called "LBB") for working on *The Last Black Box* courses
  - If working on your NB3, then include the "--system-site-packages" flag to use libraries (packages) already installed within your RPi (NB3's) OS.

```bash
mkdir _tmp
cd _tmp
python -m venv LBB --system-site-packages
```

- Activate the virtual environment
  - *Note*: You will have to do this each time you want to use your custom Python installation. However, you can get VSCode to automatically activate it for you each time you try to run python.

```bash
# From repo root
source _tmp/LBB/bin/activate
```

## Install useful packages

```bash
# Debian Linux (APT)
sudo apt install python3-dev build-essential # might be required
sudo apt install portaudio19-dev # required for 64-bit pyaudio build

# Python (PIP)
pip install --upgrade pip
pip install numpy matplotlib
pip install setuptools wheel
pip install pyaudio wave
```

## Add local (LBB) libray paths
You can include custom Python libraries by adding a ".pth" file to the *site-packages* folder with the absolute path to your library.

```bash
# Insert the path (first bit of text) into (>) a *.pth file contained in your LBB virtual environment

# On Host (current Python version 3.12.3)
echo "/home/${USER}/NoBlackBoxes/LastBlackBox/boxes/audio/python/libs" >> /home/${USER}/NoBlackBoxes/LastBlackBox/_tmp/LBB/lib/python3.12/site-packages/NB3.pth

# On NB3 (current Python version 3.11.2)
echo "/home/${USER}/NoBlackBoxes/LastBlackBox/boxes/audio/python/libs" >> /home/${USER}/NoBlackBoxes/LastBlackBox/_tmp/LBB/lib/python3.11/site-packages/NB3.pth
```

---
