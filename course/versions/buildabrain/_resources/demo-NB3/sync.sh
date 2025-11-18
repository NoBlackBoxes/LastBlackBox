#!/bin/bash -i
set -eu

# Sync "The Last Black Box" GitHub repo
echo "Syncing LBB Git Repo..."
git -C ${LBB} pull

# Get the current Python version (into an environment variable that can be used by Linux)
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version is $PYTHON_VERSION"

# Insert the local libraries path (libs) into (>) a *.pth file contained in your LBB virtual environment
echo "$HOME/NoBlackBoxes/LastBlackBox/libs" > "$HOME/NoBlackBoxes/LastBlackBox/_tmp/LBB/lib/python${PYTHON_VERSION}/site-packages/local.pth"
echo "LastBlackBox python libraries (libs) folder added to search path (as local.pth)"

# Copy custom .bashrc to home directory
echo "Copying custom .bashrc..."
cp ${LBB}/course/versions/buildabrain/_resources/demo-NB3/.bashrc ${HOME}/.

# Sourcing .bashrc
echo "Sourcing .bashrc..."
source ${HOME}/.bashrc

echo "Sync Complete."
