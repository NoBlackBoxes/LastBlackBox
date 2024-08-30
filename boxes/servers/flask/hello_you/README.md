# Servers : Flask : Hello <You>

A simple Flask app, says hello to <you>

## Run

```bash
# The FLASK_APP environment variable is set, then run
FLASK_APP=hello_you.py flask run

# To run on your local network
FLASK_APP=hello_you.py flask run --host 0.0.0.0

# To enable debugging
FLASK_DEBUG=1 FLASK_APP=hello_you.py flask run --host 0.0.0.0
```

