# Servers : Flask : Hello

The simplest Flask app, says hello to anyone

## Run

```bash
# The FLASK_APP environment variable is set, then run
FLASK_APP=hello.py flask run

# To run on your local network
FLASK_APP=hello.py flask run --host 0.0.0.0

# To enable debugging
FLASK_DEBUG=1 FLASK_APP=hello.py flask run --host 0.0.0.0
```

