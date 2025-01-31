# Build a Brain : How the Internet Works
Computers talking to other computers forms a **network**. Networks talking to other networks forms the ***the internet***...assuming everyone agrees to speak the same language (which is called *HTTP*).

## Networks
Things started getting really interesting when organisms began to interact with each other. In this box, we will enter the world of **networks** and start by communicating between our local NB3's. We will then move on to communicating with the outside world, via the *internet*.

<details><summary><i>Materials</i></summary><p>

Name|Depth|Description| # |Data|Link|
:-------|:---:|:----------|:-:|:--:|:--:|
Cable (MiniUSB-20cm)|01|Short mini-USB to Type-A cable (20 cm)|1|[-D-](/boxes/networks/)|[-L-](https://www.amazon.co.uk/LINDY-0-2-Type-Mini-B-Cable/dp/B01IZ4VFCO)

</p></details><hr>

### Physical Layers
> Connections must be made between computers in order for them to communicate. These can be wires carrying electrical signals (from two wires for serial connections to many wires for parallel connections) or wireless that most often uses light (radio waves). This is the physical layer of the network...and a message may traverse many different types of physical connections on its path from the sender to the receiver.


### Sockets
> Sockets abstract the idea of a connection between computers. A socket communicates to another socket using either TCP or UDP packets. It has an address. It can listen for and request connections.


#### Watch this video: [NB3 : Develop a Protocol](https://vimeo.com/1042782602)
> Let's develop a simple network protocol for communication between our NB3's midbrain computer and hindbrain microcontroller.

- Decide on your command format (x,o)
- Run command server on your hindbrain
- Send commands from your midbrain

## Websites
Creating a website, from simple to complex.

<details><summary><i>Materials</i></summary><p>

Name|Depth|Description| # |Data|Link|
:-------|:---:|:----------|:-:|:--:|:--:|

</p></details><hr>

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

<details><summary><i>Materials</i></summary><p>

Name|Depth|Description| # |Data|Link|
:-------|:---:|:----------|:-:|:--:|:--:|

</p></details><hr>

### HTTP
> The common language of the internet is HTTP...the Hyper Text Transfer Protocol, which delivers the HTML files to your browser.


### NB3 : Host a Website
> Host a simple website on your NB3!


# Project
### NB3 : Remote Control NB3 (TUI)
> Let's remotely control our NB3 using a simple text-based user interface (TUI). We will detect a keypress in the terminal and send the appropriate command to our hindbrain motor controller.

<details><summary><weak>Guide</weak></summary>
:-:-: A video guide to completing this project can be viewed <a href="https://vimeo.com/1042784651" target="_blank" rel="noopener noreferrer">here</a>.
</details><hr>

- Expand your hindbrain command repertoire
- Detect a key press in Python (on your midbrain)
- Send a command to your hindbrain using the serial connection
- Your goal is to press a key on your "host" computer and have your NB3 respond. If you detect different keys, then you can have your NB3 respond with different behaviors/directions.
- This is a ***creative*** task with lots of different solutions. However, to get you started, I have created the example code described below.
- SSH connection from your "host" computer to your NB3.
- Code to detect keypresses with your NB3's Raspberry Pi (keypresses are sent via SSH whenever you type in the terminal window)
- Python example code for detecting keypresses: [python keyboard](/boxes/python/remote-NB3/keyboard/keyboard.py)
- Code to send "serial" commands from your NB3's midbrain (RPi) to hindbrain (Arduino)
- Python example code to send serial commands: [python serial](/boxes/python/remote-NB3/serial/serial_write.py)
- Code to run on your NB3's hindbrain (Arduino) that listens for serial commands and responds with behaviour
- Arduino example code to respond to a single serial command with LED: [arduino serial server](/boxes/python/remote-NB3/arduino/serial_server/)
- Arduino example code to respond to a multiple serial command with different servo movements: [arduino serial controller](/boxes/python/remote-NB3/arduino/serial_controller/)
- Code that combines detecting keypresses and sending serial commands
- Python example code that combines keypress detection and serial command writing: [python kerial](/boxes/python/remote-NB3/kerial/kerial.py)
- Python example code that combines keypress detection (using a more capable library, **sshkeyboard**, that also detects when a key is held down) and serial command writing: [python drive](/boxes/python/remote-NB3/drive/drive.py)

