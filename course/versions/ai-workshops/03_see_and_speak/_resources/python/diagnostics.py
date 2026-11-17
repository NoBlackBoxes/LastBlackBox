"""Useful tracebacks without exposing environment credentials or SDK headers."""
import logging
import os
from pathlib import Path
import re
import traceback


def report_exception(context, error):
    details = ''.join(traceback.format_exception(error))
    for name, value in os.environ.items():
        if value and any(word in name.upper() for word in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')):
            details = details.replace(value, '[REDACTED]')
    details = re.sub(r'(?i)(Bearer\s+)\S+', r'\1[REDACTED]', details)
    details = re.sub(r'(?i)((?:xi-api-key|authorization|api[_-]?key)[\"\x27]?\s*[:=]\s*)[^\s,}]+',
                     r'\1[REDACTED]', details)
    logging.getLogger('demo3').error('%s\n%s', context, details.rstrip())
    path = Path(__file__).parent / 'last_error.log'
    try:
        path.write_text(context + '\n' + details)
    except OSError:
        logging.getLogger('demo3').warning('Could not write %s', path)
