# Bootcamp : The Internet
Computers talking to other computers forms a **network**. Networks talking to other networks forms the ***the internet***...assuming everyone agrees to speak the same language (which is called *HTTP*).

<details><summary><i>Materials</i></summary><p>

Name|Description| # |Package|Data|Link|
:-------|:----------|:-----:|:-:|:--:|:--:|
Cable (MiniUSB-20cm)|Short mini-USB to Type-A cable (20 cm)|1|Cables (001)|[-D-](/boxes/networks/)|[-L-](https://www.amazon.co.uk/gp/product/B07FW69HNT)

</p></details><hr>

## Networks
#### Watch this video: [Physical Layers](https://vimeo.com/1127178041)
<p align="center">
<a href="https://vimeo.com/1127178041" title="Control+Click to watch in new tab"><img src="../../../../boxes/networks/_resources/lessons/thumbnails/Physical-Layers.gif" alt="Physical Layers" width="480"/></a>
</p>

> Connections must be made between computers in order for them to communicate. These can be wires carrying electrical signals (from just one or two wires for serial connections to many wires for parallel connections) or wireless (most often using light, i.e. radio waves). This is the physical layer of the network...and a message may traverse many different types of physical connections on its path from the sender to the receiver.


#### Watch this video: [NB3 : Develop a Protocol](https://vimeo.com/1042782602)
<p align="center">
<a href="https://vimeo.com/1042782602" title="Control+Click to watch in new tab"><img src="../../../../boxes/networks/_resources/lessons/thumbnails/NB3_Develop-a-Protocol.gif" alt="NB3 : Develop a Protocol" width="480"/></a>
</p>

> Let's develop a simple network protocol for communication between our NB3's midbrain computer and hindbrain microcontroller.

- Decide on your command format (x,o)
- Run command server on your hindbrain
- Send commands from your midbrain

#### Watch this video: [NB3 : Remote Control NB3 (TUI)](https://vimeo.com/1042784651)
<p align="center">
<a href="https://vimeo.com/1042784651" title="Control+Click to watch in new tab"><img src="../../../../boxes/networks/_resources/lessons/thumbnails/NB3_Remote-Control-NB3-TUI.gif" alt="NB3 : Remote Control NB3 (TUI)" width="480"/></a>
</p>

> Let's remotely control our NB3 using a simple text-based user interface (TUI). We will detect a keypress in the terminal and send the appropriate command to our hindbrain motor controller.

- **GOAL**: Press a key on your PC and have your NB3 respond. If you detect different keys, then you can have your NB3 respond with different behaviors/directions.
- This is a ***creative*** task with lots of different solutions. However, to get you started, we have created the example code described below.
- Detect a key press in Python (on your midbrain)
  - Python example code for detecting keypresses: [python keyboard](/boxes/networks/remote-NB3/python/keyboard/keyboard.py)
- Send a command to your hindbrain using the serial connection
  - Python example code to send serial commands: [python serial](/boxes/networks/serial_protocol/python/serial_blink/serial_blink.py)
  - Arduino example code to respond to a single serial command with LED: [arduino serial server](/boxes/networks/serial_protocol/arduino/serial_server/serial_server.ino)
- Python example code that combines keypress detection and serial command writing: [python kerial](/boxes/networks/remote-NB3/python/kerial/kerial.py)
  - Arduino example code to respond to a multiple serial command with different servo movements: [arduino serial controller](/boxes/networks/remote-NB3/arduino/serial_controller/serial_controller.ino)
- Python example code that combines keypress detection (using a more capable library, **sshkeyboard**, that also detects when a key is held down) and serial command writing: [python drive](/boxes/networks/remote-NB3/python/drive/drive.py)
- **HAVE FUN!**

#### Watch this video: [Architectures](https://vimeo.com/manage/videos/1127222969)
<p align="center">
<a href="https://vimeo.com/manage/videos/1127222969" title="Control+Click to watch in new tab"><img src="../../../../boxes/networks/_resources/lessons/thumbnails/Architectures.gif" alt="Architectures" width="480"/></a>
</p>

> The arrangement of connections between computers (nodes) defines the network's *architecture*. These can be simple 1-to-1 connections or much more complex. Here we will introduce the architecture of our most important network, **the internet**.


#### Watch this video: [Sockets](https://vimeo.com/1134201413)
<p align="center">
<a href="https://vimeo.com/1134201413" title="Control+Click to watch in new tab"><img src="../../../../boxes/networks/_resources/lessons/thumbnails/Sockets.gif" alt="Sockets" width="480"/></a>
</p>

> Sockets abstract the idea of a connection between computers. A socket communicates to another socket using either TCP or UDP packets. It has an address. It can listen for and request connections.

- Load some new code onto your Arduino that streams the analog voltage values measured at pin A0 (scaled 0 to 255) over the serial connection to your RPi. Feel free to connect something interesting, like a light sensor, to the AO input. Here is some example Arduino code: [Analog Stream](/boxes/networks/sockets/arduino/analog_stream/analog_stream.ino)
- Run the "socket server" Python example on your NB3's RPi. [Socket Server](/boxes/networks/sockets/python/server/socket_server.py)
  - This code waits for a "client" to connect and then starts sending the serial data received from Arduino over to the connected socket. The data is sent in small chunks (buffers), default size is 16 bytes...but you can change this in the code and see what happens to the streaming latency/performance.
- Run the "socket client" Python example on your PC (you will need a new VS Code window). [Socket Client](/boxes/networks/sockets/python/client/socket_client.py)
  - This code will form a connection to the "socket server" running on your NB3 and receive the data it sends. It prints the data received to the terminal.
  - If you prefer a cool "real-time plot", then you can use this (very simplistic) plotting library (NB3.Plot) to open a window on your PC and view a simple line plot of the data received from the socket. [Socket Client Pyglet Plot](/boxes/networks/sockets/python/client/socket_client_pyglet.py)
    - Note: This plotting code uses the Python "pyglet" library, a very nice wrapper of OpenGL (graphics library). It gives you *a lot* of control of the graphics processing on your PC and can produce some beautiful, fast visualizations of real-time data (and can even be used to make games!). We encourage you to explore and have fun!

## Websites
#### Watch this video: [HTML](https://vimeo.com/1134724080)
<p align="center">
<a href="https://vimeo.com/1134724080" title="Control+Click to watch in new tab"><img src="../../../../boxes/websites/_resources/lessons/thumbnails/HTML.gif" alt="HTML" width="480"/></a>
</p>

> Hyper-text markup language (HTML) is the data format of the world-wide-web. When you "visit a website" you are simply asking for an HTML file. What is an HTML file? It is just a text file with some special "<tags>" that control how the contents of the file should be displayed. The program you use to request these and display these files is called a *web browser*.

- Create an HTML file. You can start from the simple template here: [HTML (simple)](/boxes/websites/html/simple.html)
  - You should do this on your PC (not NB3) in a VS Code window (or any text editor)
- Open your HTML file in a browser on your PC. Do you see your awesome website?
- Add some more complex layout to your website. Here is an example: [HTML (layout)](/boxes/websites/html/layout.html)
- Find a cool website on the internet. Use the developer tools in your internet browser to "explore" the mess of HTML that underlies this website...and maybe make some creative changes.

#### Watch this video: [CSS](https://vimeo.com/1134720724)
<p align="center">
<a href="https://vimeo.com/1134720724" title="Control+Click to watch in new tab"><img src="../../../../boxes/websites/_resources/lessons/thumbnails/CSS.gif" alt="CSS" width="480"/></a>
</p>

> Webpages often have a consistent "style" across an entire website. It is possible to create and maintain this style by directly including the same style tags (colors, fonts, margins, etc.) in every HTML page, but this is difficult to maintain and tedious to update. Cascading Style Sheets (CSS) allow you to put all of your "styling" into a single file that can be included on the pages that need it, giving you one place to make tweaks to the styling of your entire website.

- Create a "styles.css" for your website. Add a link to this "stylesheet" in your HTML code. You can start with the examples (HTML and CSS) here: [CSS Examples](/boxes/websites/css/)
- Make your site look cool, and *consistent* across different pages.

#### Watch this video: [Javascript](https://vimeo.com/1134729863)
<p align="center">
<a href="https://vimeo.com/1134729863" title="Control+Click to watch in new tab"><img src="../../../../boxes/websites/_resources/lessons/thumbnails/Javascript.gif" alt="Javascript" width="480"/></a>
</p>

> Javascript is the web browser's programming (scripting) language. It has very little to do with Java, but it is the main way to add "interaction" to websites. Everyone should know a little bit of Javascript. Also, given that everyone has access to a web browser, it is one of the easiest programming environments to setup and start playing around.

- Add some (Javascript) code to your website to make it more interactive.
- Here is an example to respond to a "button press" by hiding/revealing the navigation bar in a simple website: [Button Response](/boxes/websites/javascript/alternates/toggle.js)
- Here is another example to gets the current time and updates a live clock page: [Live Clock](/boxes/websites/javascript/alternates/clock.js)
- There is way more you can do with Javascript. Have fun exploring and learning!

## Servers
#### Watch this video: [HTTP](https://vimeo.com/1135853103)
<p align="center">
<a href="https://vimeo.com/1135853103" title="Control+Click to watch in new tab"><img src="../../../../boxes/servers/_resources/lessons/thumbnails/HTTP.gif" alt="HTTP" width="480"/></a>
</p>

> The fundamental "application protocol" of the internet is HTTP (Hyper Text Transfer Protocol). Your web browser (the client) uses HTTP to *request* HTML files (and other resources) from a computer that understands HTTP (the server) anywhere on the internet. Here we will build a *very simple* HTTP server in Python.

- The code for the "simplest" HTTP socket server is [here](/boxes/servers/python/simple_http_server/simplest_http_server.py)
  - Run this code and confirm that it is only able to serve a single file [index.html](/boxes/servers/python/simple_http_server/index.html)
- The code for a "simple" HTTP socket server (capable of serving other HTML files) is [here](/boxes/servers/python/simple_http_server/simple_http_server.py)
  - Run this code and confirm that the "link" to another HTML file works correctly.
- **BONUS**: Try connecting to the simple HTTP server running on your NB3 from the web browser on your phone (you must be on the same local network, e.g. same WiFi)

#### Watch this video: [NB3 : Host a Website](https://vimeo.com/1135859914)
<p align="center">
<a href="https://vimeo.com/1135859914" title="Control+Click to watch in new tab"><img src="../../../../boxes/servers/_resources/lessons/thumbnails/NB3_Host-a-Website.gif" alt="NB3 : Host a Website" width="480"/></a>
</p>

> Let's host a website on your robot using the NB3 Server library and Python. We will first explain how the NB3 (HTTP) Server class works, what it is capable of, and then host a site with a range of different file types: HTML, CSS, images, icons, and Javascript.

- Explore the Python code for the [NB3 Server class](/libs/NB3/Server/server.py).
- Run the [NB3 Server Test](/boxes/servers/python/NB3_server_example/test_NB3_server.py) Python code on your robot.
  - The website resources "served" by this example are [here](/boxes/servers/python/NB3_server_example/site/).
- **TASK**: Challenge the NB3 Server's multi-threading.
  - While the server is running on your NB3, try connecting to it with multiple different web browsers (your PC, phone, tablet, etc.). This will test the "multi-threaded" handling of multiple simultaneous connections.
  - **BONUS**: Look up DDOS (Distributed Denial-of-Service) attacks. Can you use this approach to find the limits of the NB3 Server?
> The server should respond appropriately and each device should receive the same website. If it doesn't...please let us know!

# Project
#### Watch this video: [NB3 : Remote Control NB3 (GUI)](https://vimeo.com/1135870296)
<p align="center">
<a href="https://vimeo.com/1135870296" title="Control+Click to watch in new tab"><img src="../../../../boxes/servers/_resources/lessons/thumbnails/NB3_Remote-Control-NB3-GUI.gif" alt="NB3 : Remote Control NB3 (GUI)" width="480"/></a>
</p>

> Let's remotely control your NB3 using a graphical user interface (GUI) that you can access from *any* web browser.

- **GOAL**: Host a website on your NB3 that has interactive buttons that you click (or touch on a touchscreen). Each button will generate different motions of your robot (forward, backward, left, right, etc.).
- This is a ***creative*** task with lots of different solutions. However, to get you started, we have created the example code described below.
- Upload the [serial controller](/boxes/networks/remote-NB3/arduino/serial_controller/serial_controller.ino) code to your **Arduino** hindbrain.
  - *Optional*: Expand your hindbrain command repertoire to include LED blinks, buzzes...the choice is yours.
- Host a website with interactive buttons by running the [Remote Control GUI](/boxes/servers/python/remote-NB3_GUI/remote_control_GUI.py) server code on your NB3.
  - The website's HTML, CSS, and Javascript is [here](/boxes/servers/python/remote-NB3_GUI/site/).
- **TASK**: Add a behaviour for the "Do Action" button.
  - This can be a simple sequence of movements (using commands already understood by your hindbrain), or you can create an entirely new behaviour for either your hindbrain or midbrain to execute.
> Your "Action" button" (?) should now do something cool!

