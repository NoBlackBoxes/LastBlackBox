# The Last Black Box Bootcamp: Day 2 - Computers

## Morning

----

### Logic and Memory

- MOSFET Gates and floating gate MOSFET memory

- Computers!

### NB3 Build (hindbrain)

- Add an Arduino as your NB3 Hindbrain

<p align="center">
<img src="resources/images/pinout_arduino_nano.png" alt="arduino pinout" width="650" height="700">
</p>

- Install the Arduino IDE

  - Board: Arduino Nano
  - Processor: ATmeag328P
  - Port: *this is unique to your setup and depends on your operating system*

- Watch this video: [NB3 Hindbrain](https://vimeo.com/626836554)
  - This pinout for the Arduino Nano might be useful: [Anduino Nano Pinout](resources/images/pinout_arduino_nano_clone.png)
- ***Task 2***: Mount and power your Arduino based hindbrain (connect the mini-USB cable)
  - The built-in LED on the board should be blinking at 1 Hz

### Programming Arduino

- If you are new to programming microcontrollers (or programming in general), then watch this video before starting the programming tasks: [Programming Arduino](https://vimeo.com/627783660)

- ***Task 3***: Blink an (external) LED 
  - Connect the LED to a digital output pin (D13 in the example below), but don't forget your current limiting resistor!

<p align="center">
<img src="resources/images/LED_driver_circuit.png" alt="LED driver" width="400" height="300">
</p>

- ***Task 4***: Measure an analog signal from your LDR light sensor circuit
  - Send the output voltage of your light sensor (the "middle" of the divider) to an analog input pin.
  - Check out the example in (*File->Examples->Basic->AnalogReadSerial*) to see how to use the "Serial Monitor" to report the analog voltage signal measured from your light sensor back to your host computer.
  - Write a program that will turn on your LED (from *Task 2*) when the light signal is above (or below) some threshold.
- ***Task 5***: Generate a *pulsing* signal for your piezo buzzer
  - The piezo buzzer will expand (5V) and contract (0V) as you switch the voltage applied accross it. This expansion/contraction forces air into/out of the plastic case. If you switch it ON/OFF fast enough, then you can *hear it*!
    - Use the "Blink" example...but with a much shorter delay between the ON/OFF "blinks". How short until you can hear something?
    - *Note*: make sure the tiny wire ends of the buzzer cables are firmly connected your digital output pin (red) and ground (black). You can also use your aligator clips if it is too difficult to connect them to the breadboard.
  - Now investigate Arduino's "tone" Library. Some examples can be found in the *File->Examples-Digital* folder. You can use this library to make some (slightly) more pleasant sounds.
  - This is a piezo buzzer:

<p align="center">
<img src="resources/images/piezo_buzzer.png" alt="Piezo Buzzer" width="400" height="300">
</p>
 
- ***Task 6***: Build a Theremin
  - What if you used the analog voltage signal measured from your light sensor to change the frequency of the "tone" playing on your buzzer? Hmm...
- ***Task 7***: ***Have fun!***

----

## Afternoon

----

### Behaviour (and programming)

### NB3 Build (servos)

- Watch this video: [LBB Servos](https://vimeo.com/843653329)
- Watch this video: [NB3 Servos](https://vimeo.com/843664157)
- ***Task 1***: Mount the robot servo motors, wheels, and caster (ball bearing)

- ***Project***: Extend your robot's behaviour (be creative!)

*Suggestion*: Try building a Braitenberg vehicle. The servo test code in today's [resources/arduino](resources/arduino/servo_test) folder will help you get your motors moving. Can you make there speed dependent on how bright it is on the left or right side of your NB33 (you will need *two* light sensors?

<p align="center">
<img src="resources/images/braitenberg_vehicle.png" alt="Braitenberg Vehicle" width="600" height="300">
</p>

----
