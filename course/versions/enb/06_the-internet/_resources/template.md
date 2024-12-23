# Bootcamp : The Internet

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
