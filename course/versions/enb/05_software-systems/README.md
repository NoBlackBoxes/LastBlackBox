# Bootcamp : Session 3 - Software

---
# Morning

## Power
Running more capable software requires a faster computer, which requires more power. We will first explore how a power supply works and then install one on your NB3.

#### Watch this video: [NB3-Power](https://vimeo.com/1005162740)
> We install a DC-DC power supply on our NB3.

- **TASK**: Add a (regulated) 5 volt power supply to your robot, which you can use while debugging to save your AA batteries and to provide enough power for the Raspberry Pi computer.
- *NOTE*: Your NB3_power board cable *might* have inverted colors (black to +5V, red to 0V) relative to that shown in the assembly video. This doesn't matter, as the plugs will only work in one orientation and the correct voltage is conveyed to the correct position on the body.
<details><summary><strong>Target</strong></summary>
Your NB3 should now look like this: <p align="center">
<img src="../../../boxes/power/_resources/images/NB3_power_wiring.png" alt="NB3 power wiring" width="400">
</p>
</details><hr>

## Systems
Modern computers combine a huge number of different technologies into a functional "system". They still need a core CPU and memory (RAM), but also a graphics processor, a network connection (wired and wireless), specialized hardware

#### Watch this video: [NB3 Midbrain](https://vimeo.com/1005170402)
> Add a more powerful midbrain computer (Raspberry Pi) to your NB3.

- **TASK**: Mount a Raspberry Pi on your robot (and connect its power inputs using your *shortest* jumper cables, 2x5V and 2x0V from the NB3, to the correct GPIO pins on the RPi...please *double-check* the pin numbers)
- This pinout of the Raspberry Pi GPIO might be useful: [Raspberry Pi GPIO](../../../boxes/systems/_resources/images/rpi_GPIO_pinout.png)

#### Watch this video: [RPiOS](https://vimeo.com/??????)
> After mounting and wiring your NB3's midbrain computer, you must now give it some core software to run...an operating system.
  
- **TASK**: Install the Linux-based Raspberry Pi OS on your NB3
- Follow these [RPiOS installation instructions](../../../boxes/systems/rpios/README.md)

#### Watch this video: [Connecting to RPi](https://vimeo.com/??????)
> When you have installed your NB3's operating system, then you can power it on and try to connect to it from your Host computer over WiFi.

- ***TASK***: Connect to your NB3 via WiFi
- Follow these instruction [Connecting to RPi](../../../boxes/systems/connecting/README.md)

## Linux
A free and open source operating system.

### Watch this video: [Linux: Introduction](https://vimeo.com/1005196173)
> Linux us based on UNIX.

- **TASK**: Explore Linux. Spend any extra time you have fiddling, playing with the UNIX approach to controlling a computer. Create some folders. Edit some files.

### Watch this video: [Git](https://vimeo.com/??????)
> Git is a program that keeps track of changes to your files. It is very useful when developing code. This entire course is stores as a git "repository".

- **TASK**: "Clone" (copy) all of the code in the LastBlackBox GitHub repository directly to your NB3's midbrain. It will help with later exercises if we all put this example code at the same location on the Raspberry Pi (the "home" directory).

```bash
cd ~                # Navigate to "home" directory
mkdir NoBlackBoxes  # Create NoBlackBoxes directory
cd NoBlackBoxes     # Change to NoBlackBoxes directory

# Clone LBB repo (only the most recent version)
git clone --depth 1 https://github.com/NoBlackBoxes/LastBlackBox
```

## Python
Python is an interpreted programming language.

### Watch this video: [Virtual Environments](https://vimeo.com/??????)
> We will next create a Python virtual environment on our NB3 that will isolate the specific software packages we require for the course from the Python packages used by Raspberry Pi's operating system.

- **TASK**: Create a "virtual environment" called LBB
- Follow the instructions here: [virtual environments](../../../boxes/python/virtual_environments/README.md)

---
# Afternoon

## Networks

## Websites

## Servers

---

# Project
### Build a remote-control robot!
Your goal is to press a key on your "host" computer and have your NB3 respond. If you detect different keys, then you can have your NB3 respond with different behaviours/directions.

This is a ***creative*** task with lots of different solutions. However, to get you started, I have created the example code described below.

0. SSH connection from your "host" computer to your NB3.
1. Code to detect keypresses with your NB3's Raspberry Pi (keypresses are sent via SSH whenever you type in the terminal window)
   - Python example code for detecting keypresses: [python keyboard](../../../boxes/python/remote-NB3/keyboard/keyboard.py)
2. Code to send "serial" commands from your NB3's midbrain (RPi) to hindbrain (Arduino)
    - Python example code to send serial commands: [python serial](../../../boxes/python/remote-NB3/serial/serial_write.py)
3. Code to run on your NB3's hindbrain (Arduino) that listens for serial commands and responds with behaviour
    - Arduino example code to respond to a single serial command with LED: [arduino serial server](../../../boxes/python/remote-NB3/arduino/serial_server/)
    - Arduino example code to respond to a multiple serial command with different servo movements: [arduino serial controller](../../../boxes/python/remote-NB3/arduino/serial_controller/)
4. Code that combines detecting keypresses and sending serial commands
   - Python example code that combines keypress detection and serial command writing: [python kerial](../../../boxes/python/remote-NB3/kerial/kerial.py)
   - Python example code that combines keypress detection (using a more capable library, **sshkeyboard**, that also detects when a key is held down) and serial command writing: [python drive](../../../boxes/python/remote-NB3/drive/drive.py)

----
