# Systems
Modern computers combine a huge number of different technologies into a functional "system". They still need a core CPU and memory (RAM), but also a graphics processor, a network connection (wired and wireless), and specialized hardware

## [NB3 Midbrain](https://vimeo.com/1005170402)
> Add a more powerful midbrain computer (Raspberry Pi) to your NB3.

- **TASK**: Mount a Raspberry Pi on your robot (and connect its power inputs using your *shortest* jumper cables, 2x5V and 2x0V from the NB3, to the correct GPIO pins on the RPi...please *double-check* the pin numbers)
  - This pinout of the Raspberry Pi GPIO might be useful: [Raspberry Pi GPIO](_data/images/rpi_GPIO_pinout.png)
> You should now be able to power on your Raspberry Pi (via your NB3) and observe some blinking LEDs.

## [RPiOS](https://vimeo.com/??????)
> After mounting and wiring your NB3's midbrain computer, you must now give it some core software to run...an operating system.
  
- **TASK**: Install the Linux-based Raspberry Pi OS on your NB3
  - Follow these [RPiOS installation instructions](../../../boxes/systems/rpios/README.md)
> You should now have a copy of RPiOS on your SD card.

## [Connecting to RPi](https://vimeo.com/??????)
> When you have installed your NB3's operating system, then you can power it on and try to connect to it from your Host computer over WiFi.

- ***Task***: Connect to your NB3 via WiFi
  - Follow these instruction [Connecting to RPi](../../../boxes/systems/connecting/README.md)
> You should now be able to login to your NB3 from your host computer via SSH.
