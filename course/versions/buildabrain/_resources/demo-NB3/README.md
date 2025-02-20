# Build a Brain : Demo NB3
A set of tools for easily running demos of NB3's abilities

## Requirements
These demos assume you have a complete *hardware* (Hindbrain, Muscles, Power, Midbrain, Ear, Mouth, Eye, and NPU) and *software* installation on the NB3. If you have followed the course, then this should be the case (and you will have already used these demos!). However, if you have only completed the *hardware* installation, then a "clone" of the NB3's OS (RPiOS with required software libraries and drivers pre-installed) can be downloaded here: [NB3 OS Clone] (dropbox)

## Environment
The NB3's bash shell will automatically activate the LBB python virtual environment (every time you open a new terminal). The LBB environment has a number of useful python packages installed, including the NB3 libraries used in the following demos.

## Aliases
The following "aliases" are stored in the student user's *.bashrc* file. They all run python scripts stored throughout the LBB repository. These aliases can be run from anywhere.

```bash
Drive   # Run the python driving controller (for remotely-controlling NB3)
Stream  # Run the camera streaming (camera images will be streamed to the indicated website)
Listen  # Run the keyword detection model and respond to commands
Look    # Run the face tracking model and follow detected faces
```

