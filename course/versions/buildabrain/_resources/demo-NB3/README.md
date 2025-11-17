# Build a Brain : Demo NB3
A set of tools for easily running demos of NB3's abilities

## Requirements
These demos assume you have a complete *hardware* (Hindbrain, Muscles, Power, Midbrain, Ear, Mouth, Eye, and NPU) and *software* installation on the NB3. If you have followed the course, then this should be the case (and you will have already used these demos!). However, if you have only completed the *hardware* installation, then a "clone" of the NB3's OS (RPiOS with required software libraries and drivers pre-installed) can be downloaded here: [NB3 OS Clone] (dropbox)

## Environment
The NB3's bash shell has a custom "alias" to activate the LBB python virtual environment. The LBB environment contains number of useful python packages installed, including the NB3 libraries used in the following demos. You must have the virtual environment active to run the NB3 demos.

```bash
Activate
```

## Aliases
The following "aliases" are stored in the student user's *.bashrc* file. They all run python scripts that are stored throughout the LBB repository. These aliases can be run from anywhere.

```bash
Sync    # Sync with the latest LBB repo (and these bash aliases)
Drive   # Run the python driving controller (for remotely-controlling NB3)
Stream  # Run the camera streaming (camera images will be streamed to the indicated website)
Drone   # Run a web interface to both view the camera and drive the NB3
Listen  # Run the keyword detection model and respond to commands
Look    # Run the face tracking model and follow detected faces
Music   # Play some music through the speaker
```

## Installed Packages
- Linux
```bash
sudo apt install build-essential python3-dev vim hdparm sysbench speedtest-cli portaudio19-dev
```
- Python
```bash
pip install --upgrade pip setuptools wheel pip install numpy==1.26.4 scipy matplotlib opencv-python pyaudio wave soundfile sshkeyboard netifaces tflite-runtime
```

## Cloning a working NB3 SD card

```bash
sudo dd if=/dev/sda of=~/Downloads/NB3-card.img bs=1M status=progress
```
- Then use "pishrink" to create an auto-decompressing smaller image

```bash
sudo pishrink NB3-card.img NB3-card-shrunk.img
```
Install pishrink befroe running the command