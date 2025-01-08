# Build a Brain : How the Internet Works
Computers talking to other computers forms a **network**. Networks talking to other networks forms the ***the internet***...assuming everyone agrees to speak the same language (which is called *HTTP*).

## Networks
Things started getting really interesting when organisms began to interact with each other. In this box, we will enter the world of **networks** and start by communicating between our local NB3's. We will then move on to communicating with the outside world, via the *internet*.

### Physical Layers
> Connections must be made between computers in order for them to communicate. These can be wires carrying electrical signals (from two wires for serial connections to many wires for parallel connections) or wireless that most often uses light (radio waves). This is the physical layer of the network...and a message may traverse many different types of physical connections on its path from the sender to the receiver.


### Sockets
> Sockets abstract the idea of a connection between computers. A socket communicates to another socket using either TCP or UDP packets. It has an address. It can listen for and request connections.


### NB3 : Develop a Protocol
> Let's develop a simple network protocol for communication between our NB3's midbrain computer and hindbrain microcontroller.

- Decide on your command format (x,o)
- Run command server on your hindbrain
- Send commands from your midbrain

## Websites
Creating a website, from simple to complex.

### HTML
> Hyper-text markup language.


### CSS
> Cascading Style Sheets.


### Javascript
> The browser's programming (scripting) language. Has very little to do with Java.


### NB3 : Build-a-Website
> Let's create a simple (static) website using HTML and CSS...and a little javascript.


## Servers
Serving HTML files to who ever requests them (via HTTP).

### HTTP
> The common language of the internet is HTTP...the Hyper Text Transfer Protocol, which delivers the HTML files to your browser.


### NB3 : Host a Website
> Host a simple website on your NB3!


# Project
### NB3 : Remote Control NB3 (TUI)
> Let's remotely control our NB3 using a simple text-based user interface (TUI). We will detect a keypress in the terminal and send the appropriate command to our hindbrain motor controller.

- Expand your hindbrain command repertoire
- Detect a key press in Python (on your midbrain)
- Send a command to your hindbrain using the serial connection

