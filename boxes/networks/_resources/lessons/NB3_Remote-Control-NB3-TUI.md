# Networks : NB3 : Remote Control NB3 (TUI)
Let's remotely control our NB3 using a simple text-based user interface (TUI). We will detect a keypress in the terminal and send the appropriate command to our hindbrain motor controller.

## [Video](https://vimeo.com/1042784651)

## Concepts

## Connections

## Lesson
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
