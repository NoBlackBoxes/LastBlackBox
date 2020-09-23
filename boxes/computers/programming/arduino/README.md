# computers : programming : arduino

Here we will use an integrated development enviornment (IDE) designed to make programming an Arduino board easier.

## Prerequisites

Remove an existing installation:

```bash
sudo apt remove arduino
```

Install the Arduino IDE from the [Arduino](https://www.arduino.cc/en/Main/Software)
 - Download the Linux 32-bit ARM version
 - Extract

```bash
tar -xhf <arduino-download-tar-file>
```

- Install (change to extracted directory) and run script

```bash
sudo ./install.sh
```
- Run the Arduino IDE (from command line or using the RPi menu)

- Set the following under "Tools"
  - Board: Arduino Nano
  - Processor: ATmega328P
  - Port: "/dev/ttyUSB0"

- Open an example sketch ("Blink") and try to "Upload"

----
