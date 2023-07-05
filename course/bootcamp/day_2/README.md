# The Last Black Box Bootcamp: Day 2 - Computers

## Morning

----

### NB3 Build (power)

- Watch this video: [NB3 Power](https://vimeo.com/626839902)
- *Task 1*: Add a (regulated) 5 volt power supply to your robot

### Analog to Digital

- Watch this video: [ADCs](https://vimeo.com/627773247)

### NB3 Build (hindbrain)

- Watch this video: [NB3 Hindbrain](https://vimeo.com/626836554)
  - This pinout for the Arduino Nano might be useful: [Anduino Nano Pinout](resources/images/pinout_arduino_nano.png)
- *Task 1*: Mount and power your Arduino based hindbrain (connect the mini-USB/micro-USB cable)
  - The built-in LED on the board should be blinking at 1 Hz

- ***NOTE: If you are new to Arduino (or want a refresher), watch the "Programming Arduino" video before completing the following tasks.***

- *Task 2*: Blink an (external) LED 
  - Connect the LED to a digital output pin (D13 in the example below), but don't forget your current limiting resistor!

<p align="center">
<img src="resources/images/LED_driver_circuit.png" alt="LED driver" width="400" height="300">
</p>

- *Task 3*: Measure an analog signal from your LDR light sensor circuit
  - Send the output voltage of your light sensor (the "middle" of the divider) to an analog input pin.
  - Check out the example in (*File->Examples->Basic->AnalogReadSerial*) to see how to use the "Serial Monitor" to report the analog voltage signal measured from your light sensor back to your host computer.
  - Write a program that will turn on your LED (from *Task 2*) when the light signal is above (or below) some threshold.
- *Task 4*: Generate a *pulsing* signal for your piezo buzzer
  - The piezo buzzer will expand (5V) and contract (0V) as you switch the voltage applied accross it. This expansion/contraction forces air into/out of the plastic case. If you switch it ON/OFF fast enough, then you can *hear it*!
    - Use the "Blink" example...but with a much shorter delay between the ON/OFF "blinks". How short until you can hear something?
    - *Note*: make sure the tiny wire ends of the buzzer cables are firmly connected your digital output pin (red) and ground (black). You can also use your aligator clips if it is too difficult to connect them to the breadboard.
  - Now investigate Arduino's "tone" Library. Some examples can be found in the *File->Examples-Digital* folder. You can use this library to make some (slightly) more pleasant sounds.
  - This is a piezo buzzer:

<p align="center">
<img src="resources/images/piezo_buzzer.png" alt="Piezo Buzzer" width="400" height="300">
</p>
 
- *Task 5*: Build a Theremin
  - What if you used the analog voltage signal measured from your light sensor to change the frequency of the "tone" playing on your buzzer? Hmm...
- *Task 6*: ***Have fun!***

### Programming Arduino

- If you are new to programming microcontrollers (or programming in general), then watch this video before starting the programming tasks: [Programming Arduino](https://vimeo.com/627783660)

----

## Afternoon

----

### Computers

- Live Lecture: "Logic, memory, and the *programmable* computer"
- ***Project***: Extend your robot's behaviour (be creative!)

----

## *Evening*

----

### Control

In order to more accurately control the speed of your robot (or to detect when your desired speed is not the same as your actual speed, e.g. if your robot gets stuck), then you will need some ***sensory feedback*** about how fast your motor is spinning. Your motor has this ability.

  - The other 4 cables coming out of your motor (xEa, xE+, xE-, xEb, where x is either "r" or "l" depending on the side of the motor) carry *sensory* information about how fast the motor is spinning. If you apply +5V to xE+ and 0V to xE-, then xEa and xEb will each pulse from low (0V) to high (+5V) every time the motor makes one revolution.
  - *Note*: The pulse comes once for each revolution of the *motor*, not the wheel...which revoles mouch slower given the gears.
  - The two *encoders* (Ea and Eb) are mounted asymmetrically. Why? So you can tell which direction the motor is spinning by looking at which one pulses first each revolution.

<p align="center">
<img src="resources/images/motor_interface.png" alt="Motor Interface" width="500" height="200">
<p>

In order to measure the signals from xEa and xEb, you will likely want to use "interrupts" in your Arduino program. Interrupts are little sections of code that run everytime a digital event is detected (e.g. xEa going from low to high). Your Arduino Nano does not have enough interrupt capable digital pins to monitor all 4 encoders (2 left, 2 right). Therefore, it is fine to monitor a single encoder on either side to measure the speed of each motor (the time between each encoder pulse) and then just infer the direction from the motor control signal you are sending.