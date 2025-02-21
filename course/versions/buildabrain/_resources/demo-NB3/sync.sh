#!/bin/bash
set -eu

# Sync "The Last Black Box" GitHub repo
echo "Syncing LBB Git Repo..."
git -C ${LBB} pull

# Copy .bash_aliases to home directory
echo "Copying .bash_aliases..."
cp ${LBB}/course/versions/buildabrain/_resources/demo-NB3/.bash_aliases ${HOME}/.

# Sourcing .bashrc
echo "Sourcing .bashrc..."
source ${HOME}/.bashrc

echo "Sync Complete."
