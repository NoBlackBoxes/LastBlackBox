#!/bin/bash -i
set -eu

# Sync "The Last Black Box" GitHub repo
echo "Syncing LBB Git Repo..."
git -C ${LBB} pull

# Get the current Python version (into an environment variable that can be used by Linux)
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version is $PYTHON_VERSION"

# Insert the local libraries path (libs) into (>) a *.pth file contained in your LBB virtual environment
echo "/home/${USER}/NoBlackBoxes/LastBlackBox/libs" > "/home/${USER}/NoBlackBoxes/LastBlackBox/_tmp/LBB/lib/python${PYTHON_VERSION}/site-packages/local.pth"
echo "LastBlackBox python libraries (libs) folder added to search path (as local.pth)"

# Copy .bash_aliases to home directory
echo "Copying .bash_aliases..."
cp ${LBB}/course/versions/buildabrain/_resources/demo-NB3/.bash_aliases ${HOME}/.

# Sourcing .bashrc
echo "Sourcing .bashrc..."
source ${HOME}/.bashrc

echo "Sync Complete."
