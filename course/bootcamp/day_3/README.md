# The Last Black Box : *Bootcamp* : Day 3 - Systems and Networks

----------
## Morning

### Recap

> *"Logic, Memory, CPU, Microcontroller, Arduino, Programming"*

#### Power

- *Watch this video*: [NB3 Power](https://vimeo.com/1005162740)
  - ***Task***: Add a (regulated) 5 volt power supply to your robot, which you can use while debugging to save your AA batteries and to provide enough power for the Raspberry Pi computer.
    - *NOTE*: Your NB3_power board cable *might* have inverted colors (black to +5V, red to 0V) relative to that shown in the assembly video. This doesn't matter, as the plugs will only work in one orientation and the correct voltage is conveyed to the correct position on the body.

<p align="center">
<img src="../../../boxes/power/_data/images/NB3_power_wiring.png" alt="NB3 power wiring" width="400" height="300">
</p>

#### Systems

- *Watch this video*: [NB3 Midbrain](https://vimeo.com/1005170402)
  - ***Task***: Mount a Raspberry Pi on your robot (and connect its power inputs using your *shortest* jumper cables, 2x5V and 2x0V from the NB3, to the correct GPIO pins on the RPi...please *double-check* the pin numbers)
    - This pinout of the Raspberry Pi GPIO might be useful: [Raspberry Pi GPIO](../../../boxes/systems/_data/images/rpi_GPIO_pinout.png)

After mounting and wiring your NB3's midbrain computer, you must now give it some core software to run...an operating system.
  - ***Task***: Install the Linux-based Raspberry Pi OS on your NB3
    - Folow these [RPiOS installation instructions](../../../boxes/systems/rpios/README.md)

When you have installed your NB3's operating system, then you can power it on and try to connect to it from your Host computer over WiFi.
  - ***Task***: Connect to your NB3 via WiFi
    - Folow these instruction [Connecting to RPi](../../../boxes/systems/connecting/README.md)

#### Linux
- *Watch this video*: [Linux: Introduction](https://vimeo.com/1005196173)
  - ***Task***: Explore Linux. Spend any extra time you have fiddling, playing with the UNIX approach to controlling a computer.

------------
## Afternoon

### PROJECT
- Build a remote-control robot!!

