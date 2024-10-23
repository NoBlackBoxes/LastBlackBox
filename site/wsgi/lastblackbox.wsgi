#!/usr/bin/python
import os
import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/noblackboxes/lastblackbox/")
import app as application
application.secret_key = os.getenv("SECRET_KEY")