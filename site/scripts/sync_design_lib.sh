#!/bin/bash
set -eu

rsync -auv --update --delete ~/NoBlackBoxes/LastBlackBox/course/_designs/libs/Design/ ~/NoBlackBoxes/LastBlackBox/site/libs/Design/
