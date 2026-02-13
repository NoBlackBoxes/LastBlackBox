# Environment Setup Guide

This repository uses a micromamba/conda environment to manage Python dependencies.

## Quick Start

### Install Micromamba (if not already installed)

```bash
# Linux/macOS
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
./bin/micromamba shell init -s bash -p ~/micromamba

# Or use conda/miniconda if you prefer
```

### Create Environment

```bash
# Using micromamba (recommended - faster)
micromamba env create -f environment.yml

# Or using conda
conda env create -f environment.yml
```

**Note**: The environment uses only `conda-forge` channel (no Anaconda defaults) for faster setup. All packages are available from conda-forge.

### Activate Environment

```bash
# Using micromamba
micromamba activate lastblackbox

# Or using conda
conda activate lastblackbox
```

## Platform-Specific Notes

### Raspberry Pi / NB3

On Raspberry Pi systems, you'll need to install additional packages:

```bash
# Install picamera2 (system package)
sudo apt update
sudo apt install python3-picamera2

# Install spidev for SPI communication (if needed)
pip install spidev
```

### PC / Development

The environment will automatically fall back to webcam support when picamera2 is not available. No additional setup required.

## Key Dependencies

- **python=3.11**: Python version
- **numpy=1.26.***: Specifically pinned to avoid OpenCV compatibility issues with numpy 2.x
- **opencv**: Computer vision library
- **pyaudio**: Audio I/O (via conda-forge)
- **librosa**: Audio analysis library
- **matplotlib, scipy, pandas**: Scientific computing
- **flask**: Web framework
- **pyglet**: Graphics/visualization
- **netifaces, psutil**: System/networking utilities

### Optional Packages (Not Included by Default)

For faster environment creation, heavy packages are excluded. Install separately if needed:

```bash
# Machine Learning (pytorch + torchvision ~1-2GB)
micromamba install pytorch torchvision scikit-image -c conda-forge

# CAD/3D modeling
micromamba install cadquery -c conda-forge
```

## Troubleshooting

### OpenCV Issues

If you encounter issues with OpenCV, ensure numpy is pinned to 1.26:

```bash
micromamba install numpy=1.26.*
```

### Audio Issues (pyaudio)

`pyaudio` is installed via conda-forge, which should handle dependencies automatically. If you encounter issues, you can try installing system libraries:

```bash
sudo apt install portaudio19-dev
```

### Graphics Issues (pyglet)

On Linux, ensure OpenGL libraries are installed:

```bash
sudo apt install python3-opengl
```

## Updating the Environment

```bash
micromamba env update -f environment.yml --prune
```
