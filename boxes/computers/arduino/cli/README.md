# computers : arduino : cli
Using the Arduino command line interface to upload code to the NB3's hindbrain.

## Installation (PC)
- Install gcc-avr
- Install avrdude
- Install arduino-cli

## Installation (NB3)
- Install the Arduino CLI

```bash
cd $LBB/_tmp # Change to LBB tmp folder
mkdir arduino && cd arduino # Make and enter an Arduino directory

# Download and install arduino-cli in local bin folder
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh

# Add arduino-cli to PATH (so you can run it from anywhere)
echo 'export PATH="$LBB/_tmp/arduino/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
```

- Install required AVR boards and libraries

```bash
arduino-cli core update-index
arduino-cli core install arduino:avr
arduino-cli lib install servo
```

## Upload Arduino Code (example)

```bash
# Compile
arduino-cli compile --fqbn arduino:avr:nano "name of code folder"

# Upload
arduino-cli upload -p /dev/ttyUSB0 -v --fqbn arduino:avr:nano "name of code folder"
```