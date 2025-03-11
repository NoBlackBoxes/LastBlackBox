# -----------
# NBB Student
# -----------

# Export environment variables
export LBB="${HOME}/NoBlackBoxes/LastBlackBox"

# Activate LBB python virtual environment
source ${HOME}/NoBlackBoxes/LastBlackBox/_tmp/LBB/bin/activate

# Demo Aliases
alias Sync="${LBB}/course/versions/buildabrain/_resources/demo-NB3/sync.sh"
alias Drive="python ${LBB}/boxes/networks/remote-NB3/python/drive/drive.py"
alias Stream="python ${LBB}/boxes/vision/stream-NB3/stream.py"
alias Drone="python ${LBB}/boxes/vision/drone-NB3/drone.py"
alias Listen="python ${LBB}/boxes/intelligence/NPU/listen-NB3/listen.py"
alias Look="python ${LBB}/boxes/intelligence/NPU/look-NB3/look.py"
alias Music="python ${LBB}/boxes/audio/python/output/play_wav.py ${LBB}/_tmp/sounds/Rose_Mars_APT.wav"
