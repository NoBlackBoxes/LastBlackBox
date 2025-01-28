# computers : programming : assembly

Here we will start telling our computer what to do using Assmebly language.

## Prerequisites

### Windows

Hmmm...currently non-trivial. Look for VS code extensions.

### Linux (Debian-based)

We require an "assmebler" that can convert (assemble) our text into machine code. For this, we will install the gcc-avr package.

```bash
sudo apt-get install gcc-avr
```

We will also need a tool for transferring our binary (machine) code to the Atmel microcontroller. This tool is called "avr dude".

```bash
sudo apt-get install avrdude
```

Connect your Arduino Nano to your LBB computer with a mini-USB cable.

----
