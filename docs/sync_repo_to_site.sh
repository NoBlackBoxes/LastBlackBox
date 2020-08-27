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

# Sync all README.md files in repo to "docs" folder
rsync -am \
    --exclude 'admin/' \
    --exclude 'docs/' \
    --exclude '_template/' \
    --include '*/' \
    --include '*.md' \
    --include '*.png' \
    --exclude '*' \
    ${LBBREPO}/ ${LBBREPO}/docs
#FIN