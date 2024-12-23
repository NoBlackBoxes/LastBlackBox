#!/bin/bash
set -eu
rsync -auv --update --delete ~/NoBlackBoxes/LastBlackBox/course/versions/bootcamp/01_analog-electronics/ ~/NoBlackBoxes/LastBlackBox/course/versions/enb/01_analog-electronics/
rsync -auv --update --delete ~/NoBlackBoxes/LastBlackBox/course/versions/bootcamp/02_magnets-and-semiconductors/ ~/NoBlackBoxes/LastBlackBox/course/versions/enb/02_magnets-and-semiconductors/
rsync -auv --update --delete ~/NoBlackBoxes/LastBlackBox/course/versions/bootcamp/03_digital-computers/ ~/NoBlackBoxes/LastBlackBox/course/versions/enb/03_digital-computers/
rsync -auv --update --delete ~/NoBlackBoxes/LastBlackBox/course/versions/bootcamp/04_robot-control/ ~/NoBlackBoxes/LastBlackBox/course/versions/enb/04_robot-control/
rsync -auv --update --delete ~/NoBlackBoxes/LastBlackBox/course/versions/bootcamp/05_software-systems/ ~/NoBlackBoxes/LastBlackBox/course/versions/enb/05_software-systems/
rsync -auv --update --delete ~/NoBlackBoxes/LastBlackBox/course/versions/bootcamp/06_the-internet/ ~/NoBlackBoxes/LastBlackBox/course/versions/enb/06_the-internet/
