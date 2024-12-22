#!/bin/bash
set -eu
rsync -auv --update --delete ~/NoBlackBoxes/LastBlackBox/course/versions/bootcamp/session_1/ ~/NoBlackBoxes/LastBlackBox/course/versions/enb/session_1/
rsync -auv --update --delete ~/NoBlackBoxes/LastBlackBox/course/versions/bootcamp/session_2/ ~/NoBlackBoxes/LastBlackBox/course/versions/enb/session_2/
rsync -auv --update --delete ~/NoBlackBoxes/LastBlackBox/course/versions/bootcamp/session_3/ ~/NoBlackBoxes/LastBlackBox/course/versions/enb/session_3/
rsync -auv --update --delete ~/NoBlackBoxes/LastBlackBox/course/versions/bootcamp/session_4/ ~/NoBlackBoxes/LastBlackBox/course/versions/enb/session_4/
rsync -auv --update --delete ~/NoBlackBoxes/LastBlackBox/course/versions/bootcamp/session_5/ ~/NoBlackBoxes/LastBlackBox/course/versions/enb/session_5/
rsync -auv --update --delete ~/NoBlackBoxes/LastBlackBox/course/versions/bootcamp/session_6/ ~/NoBlackBoxes/LastBlackBox/course/versions/enb/session_6/
