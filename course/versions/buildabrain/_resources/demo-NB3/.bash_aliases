# -----------
# NBB Student
# -----------

# Export environment variables
export LBB="${HOME}/NoBlackBoxes/LastBlackBox"

# Activate LBB python virtual environment
source ${HOME}/NoBlackBoxes/LastBlackBox/_tmp/LBB/bin/activate

# Demo Aliases
alias Drive="python ${LBB}/boxes/networks/remote-NB3/python/drive/drive.py"
alias Stream="python ${LBB}/boxes/vision/stream-NB3/stream.py"
