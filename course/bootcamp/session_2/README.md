# Bootcamp : Session 2 - Digital Computers
It's time to learn how transistors can perform logic, store information, and work together to create a circuit that *computes*.

---
# Morning

## Computers
It may not yet seem believable, but you can build a **computer** by combining transistors in a clever way. **Let's learn how!**

#### Watch this video: [Logic](https://vimeo.com/1005128209)

#### Watch this video: [Memory](https://vimeo.com/1005137172)

#### Watch this video: [Architecture](https://vimeo.com/1005138634)

#### Watch this video: [NB3 Hindbrain](https://vimeo.com/1005064175)
We can now add a *computer* to our robot. We will begin by adding a simple microcontroller as our NB3's hindbrain. It will be responsible for controlling the "muscles" (motors) in response to commands from another (larger) computer that we will add later to the NB3 midbrain.

- [ ] **Task**: Mount and power your Arduino-based hindbrain (connect the mini-USB cable)
> **Expected Result**: The built-in LED on the board should be blinking at 1 Hz.

#### Watch this video: [Low-Level Software]()
We can control a computer by loading a list of instructions ("operations") for the computer to run into its memory. This is called "programming". We can now start writing programs for our NB3 Hindbrain (Arduino) microcontroller. However, we will use some helpful tools to make translating our program and loading it into memory much easier. **Let's write some programs for our NB3 hindbrain!**

- [ ] **Task**: Download and install the Arduino IDE (integrated development environment). Follow the instructions for your "host" computer's operating system here: [Arduino IDE](https://www.arduino.cc/en/software)
- ***IMPORTANT***: If you have trouble connecting to your Arduino from your Laptop, then it may be necessary to install the "latest" driver from FTDI for the chip that communicates over the USB cable. This is not always necessary, so please try the normal installation first. However, if you are stuck, then please checkout these [FTDI driver installation instructions](https://support.arduino.cc/hc/en-us/articles/4411305694610-Install-or-update-FTDI-drivers).
- If you are *new to programming* microcontrollers (or programming in general), then watch this video before starting the programming tasks: [Programming Arduino](https://vimeo.com/1005131993)

- [ ] **Task**: Blink an (external) LED 
- Connect the LED to a digital output pin (D13 in the example below), but don't forget your current limiting resistor!
<p align="center">
<img src="../../../boxes/computers/_data/images/LED_driver_circuit.png" alt="LED driver" width="400">
</p>

- [ ] **Task**: Measure an analog signal from your LDR light sensor circuit
- Send the output voltage of your light sensor (the "middle" of the divider) to an analog input pin.
- Check out the example in (*File->Examples->Basic->AnalogReadSerial*) to see how to use the "Serial Monitor" to report the analog voltage signal measured from your light sensor back to your host computer.
- Write a program that will turn on your LED when the light signal is above (or below) some threshold.

- [ ] **Task**: Generate a *pulsing* signal for your piezo buzzer
- The piezo buzzer will expand (5V) and contract (0V) as you switch the voltage applied accross it. This expansion/contraction forces air into/out of the plastic case. If you switch it ON/OFF fast enough, then you can *hear it*!
- Use the "Blink" example...but with a much shorter delay between the ON/OFF "blinks". How short until you can hear something?
- *Note*: make sure the tiny wire ends of the buzzer cables are firmly connected your digital output pin (red) and ground (black). You can also use your aligator clips if it is too difficult to connect them to the breadboard.
- Now investigate Arduino's "tone" Library. Some examples can be found in the *File->Examples-Digital* folder. You can use this library to make some (slightly) more pleasant sounds.
- This is a piezo buzzer:
<p align="center">
<img src="../../../boxes/computers/_data/images/piezo_buzzer.png" alt="Piezo Buzzer" width="300">
</p>
 
- [ ] **Task**: Build a Theremin
- What if you used the analog voltage signal measured from your light sensor to change the frequency of the "tone" playing on your buzzer? Hmm...
- [ ] **Task**: ***Have fun!***
- This diagram of the Arduino "pins" will defintely be useful:
<p align="center">
<img src="../../../boxes/computers/_data/images/pinout_arduino_nano.png" alt="arduino pinout" width="650">
</p>

---
# Afternoon

## Control

#### Watch this video: [Servos](https://vimeo.com/1005150863)

## Robotics
#### Watch this video: [NB3 Drive](https://vimeo.com/1005154927)

- [ ] **Task**: Mount the robot servo motors, wheels, and caster (ball bearing)

---

# PROJECT
### Extend your robot's behaviour (be creative!)
  - **Suggestion**: Try building a Braitenberg vehicle. This [servo test](../../../boxes/computers/programming/arduino/servo_test/) example will help you get your motors moving. Can you make their speed dependent on how bright it is on the *left* or *right* side of your NB3? (you will need *two* light sensors)

<p align="center">
<img src="../../../boxes/robotics/_data/images/braitenberg_vehicle.png" alt="Braitenberg Vehicle" width="600" height="300">
</p>
