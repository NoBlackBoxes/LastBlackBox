#!/bin/bash
set -eu
rsync -auv --update --delete ~/NoBlackBoxes/LastBlackBox/course/versions/bootcamp/01_electronics/ ~/NoBlackBoxes/LastBlackBox/course/versions/enb/01_session/
rsync -auv --update --delete ~/NoBlackBoxes/LastBlackBox/course/versions/bootcamp/02_magnetism_semiconductors/ ~/NoBlackBoxes/LastBlackBox/course/versions/enb/02_session/
rsync -auv --update --delete ~/NoBlackBoxes/LastBlackBox/course/versions/bootcamp/03_session/ ~/NoBlackBoxes/LastBlackBox/course/versions/enb/03_session/
rsync -auv --update --delete ~/NoBlackBoxes/LastBlackBox/course/versions/bootcamp/04_session/ ~/NoBlackBoxes/LastBlackBox/course/versions/enb/04_session/
rsync -auv --update --delete ~/NoBlackBoxes/LastBlackBox/course/versions/bootcamp/05_session/ ~/NoBlackBoxes/LastBlackBox/course/versions/enb/05_session/
rsync -auv --update --delete ~/NoBlackBoxes/LastBlackBox/course/versions/bootcamp/06_session/ ~/NoBlackBoxes/LastBlackBox/course/versions/enb/06_session/
