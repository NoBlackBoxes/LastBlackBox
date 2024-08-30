# Servers : Flask : Hello Templates

A simple Flask app, says hello using HTML templates

## Run

```bash
# The FLASK_APP environment variable is set, then run
FLASK_APP=hello_templates.py flask run

# To run on your local network
FLASK_APP=hello_templates.py flask run --host 0.0.0.0

# To enable debugging
FLASK_DEBUG=1 FLASK_APP=hello_templates.py flask run --host 0.0.0.0
```

