# Default Commands
# ----------------
[[ $- != *i* ]] && return # If not running interactively, don't do anything

alias ls='ls --color=auto'
alias grep='grep --color=auto'
PS1='[\u@\h \W]\$ '

# LBB Environment Variables
# -------------------------
export LBB="${HOME}/NoBlackBoxes/LastBlackBox"

# LBB Aliases
# -----------
alias Activate="source ${LBB}/_tmp/LBB/bin/activate" # Activate LBB python virtual environment
