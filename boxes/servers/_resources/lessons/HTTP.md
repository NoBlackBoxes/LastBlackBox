# Servers : HTTP
The fundamental "application protocol" of the internet is HTTP (Hyper Text Transfer Protocol). Your web browser (the client) uses HTTP to *request* HTML files (and other resources) from a computer that understands HTTP (the server) anywhere on the internet. Here we will build a *very simple* HTTP server in Python.

## [Video](https://vimeo.com/1135853103)

## Concepts
- Client-Server model: client makes request, and server "serves" its response to the client
- Web server (HTTP server): client requests HTML file (and other types of files), and the server responds with that resource (if available)
- Example #1: "simplest" socket http server (just serve the same HTML file)
- Example #2: "simple" but more capable socket server (handle links...)
- More capable servers can handle more types of requests. They can also handle multiple requests arriving from multiple clients at the same time. We will take a look at a more capable server in the next video.

## Lesson
- The code for the "simplest" HTTP socket server is [here](/boxes/servers/python/simple_http_server/simplest_http_server.py)
  - Run this code and confirm that it is only able to serve a single file [index.html](/boxes/servers/python/simple_http_server/index.html)
- The code for a "simple" HTTP socket server (capable of serving other HTML files) is [here](/boxes/servers/python/simple_http_server/simple_http_server.py)
  - Run this code and confirm that the "link" to another HTML file works correctly.
- **BONUS**: Try connecting to the simple HTTP server running on your NB3 from the web browser on your phone (you must be on the same local network, e.g. same WiFi)
