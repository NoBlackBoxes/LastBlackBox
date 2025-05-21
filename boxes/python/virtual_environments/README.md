# Python : Virtual Environments

Virtual environments are self-contained installations of Python. All of the packages you install and changes you make only affect this *local* environment.

---
## Create a virtual environment

- Make a sub-folder in the **repository root** called "_tmp"
  - *Note*: Anything in a "_tmp" folder is *ignored* by Git and not synced to the main repository on GitHub.
- Create a Python virtual environment (called "LBB") for working on *The Last Black Box* courses
  - If working on your NB3, then include the "--system-site-packages" flag to use libraries (packages) already installed within your RPi (NB3's) OS.

```bash
cd /home/${USER}/NoBlackBoxes/LastBlackBox  # Navigate to repository root
mkdir _tmp
cd _tmp
python -m venv LBB --system-site-packages
cd ..
```

- Activate the virtual environment
  - *Note*: You will have to do this each time you want to use your custom Python installation. However, you can get VSCode to automatically activate it for you each time you try to run python.

```bash
cd /home/${USER}/NoBlackBoxes/LastBlackBox  # Navigate to repository root
source _tmp/LBB/bin/activate
```

## Install useful packages

```bash
# Debian Linux (APT)
sudo apt install python3-dev build-essential # might be required tp build some python packages
sudo apt install portaudio19-dev # required for 64-bit pyaudio build

# Python (PIP)
pip install --upgrade pip     # Upgrade pip
pip install numpy             # Numerical processing
pip install matplotlib        # Mathematical plots
pip install setuptools        # Tools for working with Python packages
pip install wheel             # Extracting (and compressing) "wheels" (package downloads)
pip install pyaudio           # Python audio library
pip install wave              # Wave (sound) file tools
pip install sshkeyboard       # Key press and release detection via SSH 
pip install netifaces         # Discover network interface names and addresses
```

## Add local (LBB and NB3) libraries to Python paths
You can include custom Python libraries by adding a ".pth" file to the *site-packages* folder with the absolute path to your library.

```bash
# Get the current Python version (into an environment variable that can be used by Linux)
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version is $PYTHON_VERSION"

# Insert the local libraries path (libs) into (>) a *.pth file contained in your LBB virtual environment
echo "/home/${USER}/NoBlackBoxes/LastBlackBox/libs" > "/home/${USER}/NoBlackBoxes/LastBlackBox/_tmp/LBB/lib/python${PYTHON_VERSION}/site-packages/local.pth"
echo "LastBlackBox python libraries (libs) folder added to search path (as local.pth)"
```

---
