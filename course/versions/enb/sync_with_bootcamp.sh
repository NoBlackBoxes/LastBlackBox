#!/bin/bash
set -eu
rsync -auv --update --delete ~/NoBlackBoxes/LastBlackBox/course/versions/bootcamp/session_1/ ~/NoBlackBoxes/LastBlackBox/course/versions/enb/session_1/
rsync -auv --update --delete ~/NoBlackBoxes/LastBlackBox/course/versions/bootcamp/session_2/ ~/NoBlackBoxes/LastBlackBox/course/versions/enb/session_2/
rsync -auv --update --delete ~/NoBlackBoxes/LastBlackBox/course/versions/bootcamp/session_3/ ~/NoBlackBoxes/LastBlackBox/course/versions/enb/session_3/
