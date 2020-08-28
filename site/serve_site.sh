#!/bin/bash
set -e

# Check for LBBROOT environment variable
if test -z "${LBBROOT}" 
then
      echo "\$LBBROOT is not set (exiting)"
      exit 0
fi

# Set environment variables
LBBREPO=${LBBROOT}"/repo"
LBBSITE=${LBBREPO}"/site/lastblackbox.training"

# Swtich to site path
cd ${LBBSITE}

# Run python http server
python3 -m http.server

#FIN