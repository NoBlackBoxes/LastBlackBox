# Build a Brain : The Software Stack
Modern computing relies on many layers of software to coordinate increasingly powerful hardware. Today we wil dive into this "software stack" by installing a powerful open-source **operating system** called **Linux** onto a powerful mini-computer (Raspberry Pi) in your NB3's midbrain.

## Power
Running more capable software requires a faster computer, which requires more electrical power. We will now explore how power supplies work and then install one on your NB3.

#### Watch this video: [DC-DC Converters](https://vimeo.com/1035304311)
> How does efficient DC to DC conversion work? Buck and Boost.


#### Watch this video: [NB3 : Power Supply](https://vimeo.com/1035306761)
> Let's install a DC-DC power supply on our NB3.

**TASK**: Add a (regulated) 5 volt power supply to your robot, which you can use while debugging to save your AA batteries and to provide enough power for the Raspberry Pi computer.
- *NOTE*: Your NB3_power board cable *might* have inverted colors (black to +5V, red to 0V) relative to that shown in the assembly video. This doesn't matter, as the plugs will only work in one orientation and the correct voltage is conveyed to the correct position on the body.
<details><summary><strong>Target</strong></summary>
:-:-: Your NB3 should now look like this: ![NB3 power wiring:400](../../../boxes/power/_resources/images/NB3_power_wiring.png)"
</details><hr>


## Systems
Modern computers combine a huge number of different technologies into a functional "system". They still need a core CPU and memory (RAM), but also a graphics processor, a network connection (wired and wireless), and other specialized hardware. All of these hardware devices are coordinated by a sophisticated (and complex) piece of software called an *operating system*.

#### Watch this video: [NB3 : Midbrain](https://vimeo.com/1036089510)
> Now we will add a more powerful computer (Raspberry Pi) to your NB3's midbrain.

**TASK**: Mount a Raspberry Pi on your robot (and connect its power inputs using your *shortest* jumper cables, 2x5V and 2x0V from the NB3, to the correct GPIO pins on the RPi...please *double-check* the pin numbers)
- This pinout of the Raspberry Pi GPIO might be useful: [Raspberry Pi GPIO](../../../boxes/systems/_resources/images/rpi_GPIO_pinout.png)
<details><summary><strong>Target</strong></summary>
:-:-: A powered and blinking RPi midbrain.
</details><hr>


#### Watch this video: [NB3 : RPiOS](https://vimeo.com/1036095710)
> After mounting and wiring your NB3's midbrain computer, you must now give it some core software to run...an operating system.

**TASK**: Install the Linux-based Raspberry Pi OS on your NB3
- Follow these [RPiOS installation instructions](../../../boxes/systems/rpios/README.md)
<details><summary><strong>Target</strong></summary>
:-:-: Booted!
</details><hr>


#### Watch this video: [NB3 : Connecting to RPi](https://vimeo.com/1036391512)
> When you have installed your NB3's operating system, then you can power it on and try to connect to it from your Host computer over WiFi or UART.

**TASK**: Connect to your NB3 via WiFi
- Follow these instruction [Connecting to RPi](../../../boxes/systems/connecting/README.md)
<details><summary><strong>Target</strong></summary>
:-:-: Connected!
</details><hr>


## Linux
A free and open source operating system.

#### Watch this video: [Navigating the Command Line](https://vimeo.com/1036829527)
> The original user interfaces were entirely based on text. You typed commands as a line of text into your terminal console and received the result as a string of characters on the screen. Navigating this **command line** remains a useful skill, and a necessary one when working with remote machines.

**TASK**: Explore Linux. Spend any extra time you have fiddling, playing with the UNIX approach to controlling a computer. Create some folders. Edit some files.
<details><summary><strong>Target</strong></summary>
:-:-: You should see this in the command line.
</details><hr>


#### Watch this video: [Git](https://vimeo.com/1036825331)
> Git is a program that keeps track of changes to your files. It is very useful when developing code. This entire course is stored as a git "repository" on GitHub.

**TASK**: "Clone" (copy) all of the code in the LastBlackBox GitHub repository directly to your NB3's midbrain. It will help with later exercises if we all put this example code at the same location on the Raspberry Pi (the "home" directory).
- ```bash
  cd ~                # Navigate to "home" directory
  mkdir NoBlackBoxes  # Create NoBlackBoxes directory
  cd NoBlackBoxes     # Change to NoBlackBoxes directory
  # Clone LBB repo (only the most recent version)
  git clone --depth 1 https://github.com/NoBlackBoxes/LastBlackBox
  ```

<details><summary><strong>Target</strong></summary>
:-:-: You should now have a complete copy of the LBB repo on your NB3.
</details><hr>


#### Watch this video: [Package Managers](https://vimeo.com/1036834036)
> Installing and "managing" software can get complicated. Programs that help coordinate this process are called **package managers**.


## Python
Python is an interpreted programming language.

### Introducing the Interpreter
> What is Python? Where is it? How does it work? How can it work for you?

**TASK**: Say "hello world" in Python
- Print the words "Hello World" on your terminal screen
- Print the words "Hello World" on your terminal screen many, many times
- Print the words "Hello World for the {Nth} time" on your terminal screen, where "Nth" reports the iteration count, i.e. "1", "2", "3"...or (*bonus task*) "1st", "2nd", "3rd", etc.
<details><summary><strong>Target</strong></summary>
:-:-: You should see something like "Hello World for the 1st time", "Hello World for the 2nd time", etc. printed line by line in your terminal screen.
</details><hr>


### Virtual Environments
> We will next create a Python **virtual environment** on our NB3 that will isolate the specific Python packages we require for the course from the Python packages used by the Raspberry Pi's operating system.

**TASK**: Create a "virtual environment" called LBB
- Follow the instructions here: [virtual environments](../../../boxes/python/virtual_environments/README.md)
<details><summary><strong>Target</strong></summary>
:-:-: You should now have a virtual environment activated (and installed in the folder "_tmp/LBB").
</details><hr>

**TASK**: Install some useful packages using PIP
- Install numpy
- Install matplotlib
- Make a cool plot and save it to an image file
<details><summary><strong>Target</strong></summary>
:-:-: You should now hav an image of your plot saved, which you can open and view inside VS code.
</details><hr>


# Project
### NB3 : Playing With Python
> Let's see what Python can do...and get used to what it "feels like" to do stuff with Python.

**TASK**: Let's make some fun things using Python
<details><summary><strong>Target</strong></summary>
:-:-: You should have made something fun.
</details><hr>

