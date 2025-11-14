# -----------
# NBB Student
# -----------

# Default Commands
# ----------------
[[ $- != *i* ]] && return # If not running interactively, don't do anything

alias ls='ls --color=auto'
alias grep='grep --color=auto'
PS1='[\u@\h \W]\$ '

# Export environment variables
export LBB="${HOME}/NoBlackBoxes/LastBlackBox"

# LBB Aliases
# -----------
alias Activate="source ${LBB}/_tmp/LBB/bin/activate" # Activate LBB python virtual environment

# Demo Aliases
# ------------
alias Sync="${LBB}/course/versions/buildabrain/_resources/demo-NB3/sync.sh"
alias Drive="python ${LBB}/boxes/networks/remote-NB3/python/drive/drive.py"
alias Stream="python ${LBB}/boxes/vision/stream-NB3/stream.py"
alias Drone="python ${LBB}/boxes/vision/drone-NB3/drone.py"
alias Listen="python ${LBB}/boxes/intelligence/NPU/listen-NB3/listen.py"
alias Look="python ${LBB}/boxes/intelligence/NPU/look-NB3/look.py"
alias Music="python ${LBB}/boxes/audio/i2s/python/output/play_wav.py ${LBB}/_tmp/sounds/Rose_Mars_APT.wav"

# Student Aliases
# ---------------
alias MyDrone="python ${LBB}/boxes/vision/drone-NB3/my_drone.py"
alias MyListen="python ${LBB}/boxes/intelligence/NPU/listen-NB3/my_listen.py"
alias MyLook="python ${LBB}/boxes/intelligence/NPU/look-NB3/my_look.py"

# Cosmetics
# ---------
PS1='\[\e[1;32m\]\u@\[\e[0;33m\]\h\[\e[0m\]:\[\e[1;34m\]\w\[\e[0m\]\$ ' # Fancy colored prompt
