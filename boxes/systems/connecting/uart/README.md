# Systems : Connecting : UART

Your RPi has the ability to communicate via an old-school serial console (UART). We can use the USB-to-Serial converter on your microcontroller (hindbrain) to talk directly to the RPi via a terminal emulator (minicom, PuTTY, etc.). This trick requires enabling the serial console on your RPi and connecting some jumper cables between your microcontroller and RPi's GPIO pins. Follow the steps below.

## Enable the UART console on your Raspberry Pi

1. Power off your RPi
2. Remove the SD card from your RPi
3. Insert the SD card into a device reader and connect it to your host computer
4. Open the "boot" folder of the SD card
5. Edit the "config.txt" file by adding the following line somewhere in the file.
    ```bash
    enable_uart=1
    ```
6. Eject the SD card and insert it back into the RPi

## Connect your Hindbrain's USB-to-Serial converter to the Raspberry Pi UART

1. Power off your hindbrain (Arduino) - remove any connections to the Vin or 5V pins
2. Connect the RX0 pin of the Arduino to the RXD pin of the Raspberry Pi's GPIO (pin 10) using a long jumper cable
3. Connect the TX1 pin of the Arduino to the TXD pin of the Raspberry Pi's GPIO (pin 8) using a long jumper cable
4. Connect the reset pin (RST) of the Arduino to ground (GND)
5. Connect the mini-USB cable to your Arduino (which will apply power)

## Run a "Terminal Emulator" on your Host Computer

- **Linux**: Install "minicom" and start it with the following command (your port name might be different):
```bash
minicom -D /dev/ttyUSB0 -b 115200
```
- **MacOS**: Open a terminal and start the "screen" program as follows (your port name might be different):
```bash
screen /dev/ttyUSB0 115200
```

- **Windows**: Download and install the PuTTY terminal emulator.
  - Select "Serial"
  - Select Port
  - Select Baud Rate as 115200
  - Open Window

## Connect to the Raspberry Pi Linux console

1. Power on your RPi
2. Check the boot messages as they appear
3. Login with your username and password
4. **Connected!**