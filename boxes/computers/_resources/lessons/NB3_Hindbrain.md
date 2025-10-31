# Computers : NB3 : Hindbrain
We will now add a *computer* to our robot. We be using a simple microcontroller as our NB3's hindbrain. It will be responsible for controlling the "muscles" (motors) in response to commands from another (larger) computer that we will be adding later to the NB3's midbrain.

## [Video](https://vimeo.com/1033609727)

## Concepts
- Introduce Arduino (Tour)
- Pinout
- Connecting to NB3 Body
- Powering (default blink)
- Test with Arduino IDE

## Lesson

- **TASK**: Mount and power your Arduino-based hindbrain (connect the mini-USB cable)
> The built-in LED on the board should be blinking at 1 Hz.

- **TASK**: Download and install the Arduino IDE (integrated development environment).
- Follow the instructions for your "host" computer's operating system here: [Arduino IDE](https://www.arduino.cc/en/software)
- Open the "Blink" Example: File -> Examples -> Basic -> Blink
- Upload this example to your board
- ***IMPORTANT***: If you have trouble connecting to the Arduino from your Laptop, then it may be necessary to install the "latest" driver from FTDI for the chip that communicates over the USB cable. This is not always necessary, so please try the normal installation first. However, if you are stuck, then please checkout these [FTDI driver installation instructions](https://support.arduino.cc/hc/en-us/articles/4411305694610-Install-or-update-FTDI-drivers).
> You should be able to successfully compile and upload the "Blink" example (with no errors).
