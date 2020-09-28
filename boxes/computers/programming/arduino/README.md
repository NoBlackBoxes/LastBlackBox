# computers : programming : arduino

Here we will use the Arduino language and libraries, which were designed to make programming an Arduino board easier.

## Prerequisites

You will need the Arduino tools to compile and upload "sketches" to your board. You can either use the integrated development environment (IDE) or the command line integration (CLI).

### Install the Arduino IDE

Remove an existing installation:

```bash
sudo apt remove arduino
```

Install the Arduino IDE from the [Arduino](https://www.arduino.cc/en/Main/Software)
 - Download the Linux 32-bit ARM version
 - Change to the Download directory and extract

```bash
cd ~/Downloads
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

### Install the Arduino CLI

- Download the latest Arduino CLI binary for Linux 32-bit ARM here [Arduino CLI](https://arduino.github.io/arduino-cli/installation/)

 - Change to the Download directory
 - Download the Linux 32-bit ARM version (using wget)
 - Extract

```bash
cd ~/Downloads
wget https://downloads.arduino.cc/arduino-cli/arduino-cli_latest_Linux_ARMv7.tar.gz
tar -xhf <arduino-cli-download-tar-gz-file>
```

- Copy the extracted binary (arduino-cli) to a directory in your $PATH *(PATH is an environment variable that lists the directories searched when you try to run a command from the command line)*. In the following, I assume you are using "/usr/local/bin".

```bash
sudo cp arduino-cli /usr/local/bin/.
```

- Create a configuration file that will set your Arduino default values (board, port, etc.)

```bash
arduino-cli config init
```

- Update the list of available packages and cores (...boards you can compile Arduino code for)

```bash
arduino-cli core update-index
```

-- Install the platform tools for Arduino AVR (uno, nano, etc.)

```bash
arduino-cli core install arduino:avr
```

- Connect your Arduino Nano via micro USB

- Compile and example sketch (fqbn = fully qualified board name)

```bash
arduino-cli compile --fqbn arduino:avr:nano <your-example-sketch>
```

- Upload the build result to your board (-v if you want verbose output)

```bash
arduino-cli upload -v -p /dev/ttyUSB0 --fqbn arduino:avr:nano <your-example-sketch>
```

----
