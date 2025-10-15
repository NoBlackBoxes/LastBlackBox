# The Last Black Box : Networks
In this box, you will learn about networks...

## Networks
Things started getting really interesting when organisms began to interact with each other. In this box, we will enter the world of **networks** and start by communicating between our local NB3's. We will then move on to communicating with the outside world, via the *internet*.

<details><summary><i>Materials</i></summary><p>

Name|Depth|Description| # |Package|Data|Link|
:-------|:---:|:----------|:-----:|:-:|:--:|:--:|
Cable (MiniUSB-20cm)|01|Short mini-USB to Type-A cable (20 cm)|1|Cables (001)|[-D-](/boxes/networks/)|[-L-](https://www.amazon.co.uk/gp/product/B07FW69HNT)
Cable (MicroUSB-20cm)|10|Short micro-USB to Type-A cable (20 cm)|1|Cables (001)|[-D-](/boxes/networks/)|[-L-](https://www.amazon.co.uk/gp/product/B07GBS2QZ1)

</p></details><hr>

#### Watch this video: [Physical Layers](https://vimeo.com/1127178041)
<p align="center">
<a href="https://vimeo.com/1127178041" title="Control+Click to watch in new tab"><img src="../../boxes/networks/_resources/lessons/thumbnails/Physical-Layers.gif" alt="Physical Layers" width="480"/></a>
</p>

> Connections must be made between computers in order for them to communicate. These can be wires carrying electrical signals (from just one or two wires for serial connections to many wires for parallel connections) or wireless (most often using light, i.e. radio waves). This is the physical layer of the network...and a message may traverse many different types of physical connections on its path from the sender to the receiver.


#### Watch this video: [Architectures](https://vimeo.com/manage/videos/1127222969)
<p align="center">
<a href="https://vimeo.com/manage/videos/1127222969" title="Control+Click to watch in new tab"><img src="../../boxes/networks/_resources/lessons/thumbnails/Architectures.gif" alt="Architectures" width="480"/></a>
</p>

> The arrangement of connections between computers (nodes) defines the network's *architecture*. These can be simple 1-to-1 connections or much more complex. Here we will introduce the architecture of our most important network, **the internet**.


### Sockets
> Sockets abstract the idea of a connection between computers. A socket communicates to another socket using either TCP or UDP packets. It has an address. It can listen for and request connections.


# Project
### NB3 : Develop a Protocol
> Let's develop a simple network protocol for communication between our NB3's midbrain computer and hindbrain microcontroller.

<details><summary><weak>Guide</weak></summary>
:-:-: A video guide to completing this project can be viewed <a href="https://vimeo.com/1042782602" target="_blank" rel="noopener noreferrer">here</a>.
</details><hr>

- Decide on your command format (x,o)
- Run command server on your hindbrain
- Send commands from your midbrain

### NB3 : Remote Control NB3 (TUI)
> Let's remotely control our NB3 using a simple text-based user interface (TUI). We will detect a keypress in the terminal and send the appropriate command to our hindbrain motor controller.

<details><summary><weak>Guide</weak></summary>
:-:-: A video guide to completing this project can be viewed <a href="https://vimeo.com/1042784651" target="_blank" rel="noopener noreferrer">here</a>.
</details><hr>

- Expand your hindbrain command repertoire
- Detect a key press in Python (on your midbrain)
- Send a command to your hindbrain using the serial connection
- Your goal is to press a key on your PC and have your NB3 respond. If you detect different keys, then you can have your NB3 respond with different behaviors/directions.
- This is a ***creative*** task with lots of different solutions. However, to get you started, I have created the example code described below.
- Python example code for detecting keypresses: [python keyboard](/boxes/networks/remote-NB3/python/keyboard/keyboard.py)
- Python example code to send serial commands: [python serial](/boxes/networks/serial_protocol/python/serial_blink/serial_blink.py)
- Arduino example code to respond to a single serial command with LED: [arduino serial server](/boxes/networks/serial_protocol/arduino/serial_server/serial_server.ino)
- Arduino example code to respond to a multiple serial command with different servo movements: [arduino serial controller](/boxes/networks/remote-NB3/arduino/serial_controller/serial_controller.ino)
- Python example code that combines keypress detection and serial command writing: [python kerial](/boxes/networks/remote-NB3/python/kerial/kerial.py)
- Python example code that combines keypress detection (using a more capable library, **sshkeyboard**, that also detects when a key is held down) and serial command writing: [python drive](/boxes/networks/remote-NB3/python/drive/drive.py)

