# Intelligence : pytorch : keyword spotter
Keyword detection using PyTorch.

## Install prerequisites
This is best done on your PC (not your NB3). We will use the extra power of your PC to train the network.

```bash
# Create a temporary folder for tools and data
mkdir _tmp
cd _tmp

# Create and activate an "AI" virtual environment
python -m venv AI
source ./AI/bin/activate

# Install required Python packages
pip install numpy scipy matplotlib scikit-learn torch torchsummary ptflops

# If you will only use the CPU for training, then you can save some download time with the following
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy matplotlib scikit-learn torchsummary ptflops

# Add library paths to virtual environment
LBB="/home/${USER}/NoBlackBoxes/LastBlackBox"
PROJECT="${LBB}/boxes/intelligence/pytorch/keyword_spotter"
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "${LBB}/libs" > "${PROJECT}/_tmp/AI/lib/python${PYTHON_VERSION}/site-packages/local.pth"
echo "${PROJECT}/modules" >> "${PROJECT}/_tmp/AI/lib/python${PYTHON_VERSION}/site-packages/local.pth"
```

## Download dataset
From the project folder, execute the following.

```bash
cd _tmp
mkdir -p dataset
cd _tmp/dataset
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar xvf speech_commands_v0.02.tar.gz
rm speech_commands_v0.02.tar.gz
```

## Extract features from the dataset
