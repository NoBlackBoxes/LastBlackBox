# The Last Black Box Bootcamp: Day 4 - Websites, Python, Servers, and beyond

## Morning

----

### Websites

This is a website. Just a text file written in the hyper-text markup language (HTML) that your "web browser" knows how to display in a nicely formatted way.

```html
<!DOCTYPE html>
<html>

<body>
This is a <b>website</b>
</body>

</html>
```

- ***Task 1***: Make your own website (an HTML file). Open it in your web browser.

### Servers

A server is just a computer that you can connect to and ask it to send (serve) you a file (e.g. an HTML file).

Your "web browser" is a program for asking for HTML files from servers (using the the hyper-text transfer protocol, HTTP) and then displaying them in a pretty manner.

Let's write our *own* HTTP server and run it on our Raspberry Pi (NB3)!

- ***Task 1***: Run your own server (Run this python code on your NB3, put the [index.html](resources/index.html) in the same folder as your python file).

```python
# Simple HTTP server
import os
import socket

# Load your HTML page (index.html)
path = 'index.html'
file = open(path,"r")
html = file.read()
file.close()

# Set host address (your NB3's IP) and an unused port (1234)
HOST, PORT = '', 1234

# Open a "listening" socket (waits for external connections)
listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listen_socket.bind((HOST, PORT))
listen_socket.listen(1)
print(f'Serving HTTP on port {PORT} ...')

# Serve incoming connections
while True:
    # Listen for a connection
    client_connection, client_address = listen_socket.accept()

    # When a connection arrives, retrieve/decode HTTP request
    request_data = client_connection.recv(1024)
    request_text = request_data.decode('utf-8')
    print(request_text)

    # (Optional) parse request
    #   - If you want to serve different files based on the request

    # Respond to target (send the bytes of your HTML file)
    http_response = bytes(html, 'utf-8')
    client_connection.sendall(http_response)

    # Close client connection
    client_connection.close()

#FIN
```

- ***Task 2***: Use the web browser on your laptop to "navigate" to the URL (universal record locator) at your NB3's IP address and the PORT specified in the Python program (1234):

```bash
http://IP-ADDRESS:1234

# Example: http://192.168.1.157:1234 
```

***This is the internet.***

----

## Afternoon

----

### Ears

Let's let your NB3 hear. Let's add some ears (digital MEMS microphones).

- Watch this video: [NB3 Ears](https://vimeo.com/630461945)

- The instructions for installing the driver on your RPi are here: [NB3 Ear Driver Install](https://github.com/NoBlackBoxes/BlackBoxes/tree/master/audio/i2s/driver)
- Use Python to record sound from each microphone (left and right).
  - Install pyaudio library

```bash
sudo apt-get install python3-pyaudio
```
  - Check out this example script: [Python Audio](resources/python/audio/record.py)

----
