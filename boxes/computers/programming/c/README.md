# computers : programming : c

Here we will use a compiled language (C) to program our microcontroller.

## Prerequisites

We require a "compiler" that can convert (compile) our text into machine code. For this, we will install the gcc-avr package. Note: this is the same package we installed for Assembly programming, as it contains an "assembler" as well.

```bash
sudo apt-get install gcc-avr
```

We will also need a C library. This is a collection of files that we can inlcude in our program that contain the standard functions required by the C programming language.

```bash
sudo apt-get install avr-libc
```

We will also need a tool for transferring our binary (machine) code to the Atmel microcontroller. This tool is called "avr dude".

```bash
sudo apt-get install avrdude
```

Connect your Arduino Nano to your LBB computer with a mini-USB cable.

----
