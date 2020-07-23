#!/bin/bash
set -e

# Check if LBBROOT is already exported in .bashrc file, if not, append
if grep -Fxq "export LBBROOT=${HOME}/LastBlackBox" ${HOME}/.bashrc
then
    echo "LBBROOT already set in bashrc"
else
    echo "export LBBROOT=${HOME}/LastBlackBox" >> ~/.bashrc 
    echo "LBBROOT export appended to bashrc"
fi

# Check if VKROOT is already in .bashrc file, if not, append
if grep -Fxq "export VKROOT=${HOME}/Voight-Kampff" ${HOME}/.bashrc
then
    echo "VKROOT already set in bashrc"
else
    echo "export VKROOT=${HOME}/Voight-Kampff" >> ~/.bashrc 
    echo "VKROOT export appended to bashrc"
fi
#FIN