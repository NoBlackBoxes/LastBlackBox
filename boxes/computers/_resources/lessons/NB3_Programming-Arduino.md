# Computers : NB3: Programming Arduino
An introduction to programming an Arduino microcontroller.

## [Video](https://vimeo.com/1033810807)

## Concepts
- Developing a simple project with Arduino

## Connections

## Lesson
- We can now start writing *programs* for our NB3 Hindbrain microcontroller.
- **Important**: Make sure you have the Arduino IDE software installed and that you can connect (and upload code) to your Arduino via USB.

- **TASK**: Download and install the Arduino IDE (integrated development environment). Follow the instructions for your "host" computer's operating system here: [Arduino IDE](https://www.arduino.cc/en/software)
- ***IMPORTANT***: If you have trouble connecting to your Arduino from your Laptop, then it may be necessary to install the "latest" driver from FTDI for the chip that communicates over the USB cable. This is not always necessary, so please try the normal installation first. However, if you are stuck, then please checkout these [FTDI driver installation instructions](https://support.arduino.cc/hc/en-us/articles/4411305694610-Install-or-update-FTDI-drivers).
- *Help*: If you are *new to programming* microcontrollers (or programming in general), then watch this video before starting the programming tasks: [Programming Arduino](https://vimeo.com/1005131993)
> You should be able to successfully compile and upload the "Blink" example (with no errors).

- **TASK**: Blink an (external) LED 
- *Hint*: Connect the LED to a digital output pin (D13 in the example below), but don't forget your current limiting resistor!
- ![LED Driver:400](/boxes/computers/_resources/images/LED_driver_circuit.png)
> Your external LED should now be blinking at the same time as the built-in LED (both are connected to pin 13).

- **TASK**: Measure an analog signal from your LDR light sensor circuit
- *Hint*: Send the output voltage of your light sensor (the "middle" of the divider) to an analog input pin.
- *Help*: Check out the example in (*File->Examples->Basic->AnalogReadSerial*) to see how to use the "Serial Monitor" to report the analog voltage signal measured from your light sensor back to your host computer.
- *Challenge*: Write a program that will turn on your LED when the light signal is above (or below) some threshold.
> You should see values on your host laptop

- **TASK**: Generate a *pulsing* signal for your piezo buzzer
- The piezo buzzer will expand (5V) and contract (0V) as you switch the voltage applied across it. This expansion/contraction forces air into/out of the plastic case. If you switch it ON/OFF fast enough, then you can *hear it*!
- Use the "Blink" example...but with a much shorter delay between the ON/OFF "blinks". How short until you can hear something?
- *Note*: make sure the tiny wire ends of the buzzer cables are firmly connected your digital output pin (red) and ground (black). You can also use your aligator clips if it is too difficult to connect them to the breadboard.
- Now investigate Arduino's "tone" Library. Some examples can be found in the *File->Examples-Digital* folder. You can use this library to make some (slightly) more pleasant sounds.
- This is a piezo buzzer:
- ![Piezo Buzzer:300](/boxes/computers/_resources/images/piezo_buzzer.png)
> You should here a (somewhat unpleasant) sound
