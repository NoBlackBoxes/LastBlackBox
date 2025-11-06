# Networks : Sockets
Sockets abstract the idea of a connection between computers. A socket communicates to another socket using either TCP or UDP packets. It has an address. It can listen for and request connections.

## [Video]()

## Concepts
- Abstraction (software) of connection between 2 computers on a network
- Network sockets use two protocols (primarily): TCP and UDP
- TCP: every byte of data is guaranteed to arrive, error free, in the correct order
- UDP: bytes sent off to a destination, some might get lost, might arrive in wrong order (faster)
- Socket to listen for a connection (we call this the server): listen at IP address:Port
- Socket to make a connection to server (we call this the client): connect to server IP address:port
- Python/Arduino examples

## Lesson
- Load some new code onto your Arduino that streams the analog voltage values measured at pin A0 (scaled 0 to 255) over the serial connection to your RPi. Feel free to connect something interesting, like a light sensor, to the AO input. Here is some example Arduino code: [Analog Stream](/boxes/networks/sockets/arduino/analog_stream/analog_stream.ino)

- Run the "socket server" Python example on your NB3's RPi. [Socket Server](/boxes/networks/sockets/python/server/socket_server.py)
  - This code waits for a "client" to connect and then starts sending the serial data received from Arduino over to the connected socket. The data is sent in small chunks (buffers), default size is 16 bytes...but you can change this in the code and see what happens to the streaming latency/performance.

- Run the "socket client" Python example on your PC (you will need a new VS Code window). [Socket Client](/boxes/networks/sockets/python/client/socket_client.py)
  - This code will form a connection to the "socket server" running on your NB3 and receive the data it sends. It prints the data received to the terminal.
  - If you prefer a cool "real-time plot", then you can use this (very simplistic) plotting library (NB3.Plot) to open a window on your PC and view a simple line plot of the data received from the socket. [Socket Client Pyglet Plot](/boxes/networks/sockets/python/client/socket_client_pyglet.py)
    - Note: This plotting code uses the Python "pyglet" library, a very nice wrapper of OpenGL (graphics library). It gives you *a lot* of control of the graphics processing on your PC and can produce some beautiful, fast visualizations of real-time data (and can even be used to make games!). We encourage you to explore and have fun!
