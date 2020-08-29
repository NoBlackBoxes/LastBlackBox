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

# Generate key and certificate
openssl req -x509 -out "${LBBROOT}/lbb.pem" -keyout "${LBBROOT}/key.pem" \
  -newkey rsa:2048 -nodes -sha256 \
  -subj '/CN=localhost' -extensions EXT -config <( \
   printf "[dn]\nCN=localhost\n[req]\nbasicConstraints = CA:FALSE\ndistinguished_name = dn\n[EXT]\nsubjectAltName=DNS:localhost\nkeyUsage=digitalSignature\nextendedKeyUsage=serverAuth")

# Report
echo "Done"

#FIN