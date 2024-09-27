#!/bin/bash
DATE=$(date +"%Y-%m-%d_%H%M")
rpicam-still -o /home/kampff/NoBlackBoxes/LastBlackBox/_tmp/images/timelapse/$DATE.jpg

# Start Timelapse
# - Create the "timelapse" folder in the repo _tmp directory (_tmp/images/timelapse)
# - Copy this script to the new folder
# - Add "* * * * * /home/kampff/NoBlackBoxes/LastBlackBox/_tmp/images/timelapse/timelapse.sh 2>&1" using "crontab -e"
# - ...this will take a snapshot every minute
# - remove line from crontab to cancel/end timelapse acquisition
#
# Convert to Movie
# > sudo apt install ffmpeg
# > ffmpeg -r 30 -f image2 -pattern_type glob -i '*.jpg' -s 1920x1080 -vcodec libx264 timelapse.mp4
# - 30 Hz, images, match pattern (*.jpg), HD