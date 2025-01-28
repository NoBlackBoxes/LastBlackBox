# The Last Black Box : Computers
In this box, you will learn about computers...

## Computers
It may not yet seem believable, but you can build a **computer** by combining transistors in a clever way. **Let's learn how!**

<details><summary><i>Materials</i></summary><p>

Name|Depth|Description| # |Data|Link|
:-------|:---:|:----------|:-:|:--:|:--:|
Microcontroller|01|Arduino Nano (rev.3)|1|[-D-](/boxes/computers/_resources/datasheets/arduino_nano_rev3.pdf)|[-L-](https://uk.farnell.com/arduino/a000005/arduino-nano-evaluation-board/dp/1848691)
Piezo Buzzer|01|Piezoelectric speaker/transducer|1|[-D-](/boxes/computers/_resources/datasheets/piezo_buzzer.pdf)|[-L-](https://uk.farnell.com/tdk/ps1240p02bt/piezoelectric-buzzer-4khz-70dba/dp/3267212)
Cable (MiniUSB-1m)|01|Mini-USB to Type-A cable (1 m)|1|[-D-](/boxes/computers/)|[-L-](https://uk.farnell.com/molex/88732-8602/usb-cable-2-0-plug-plug-1m/dp/1221071)

</p></details><hr>

#### Watch this video: [Architecture](https://vimeo.com/1033601146)
> The basic building blocks of a computer (memory, ALU, clock, bus, and IO) have a standard arrangement (architecture) in modern systems.


### Low-Level Programming
> We can control a computer by loading a list of instructions ("operations") into its memory. This is called *programming*.


# Project
### NB3 : Hindbrain
> We will now add a *computer* to our robot. We be using a simple microcontroller as our NB3's hindbrain. It will be responsible for controlling the "muscles" (motors) in response to commands from another (larger) computer that we will be adding later to the NB3's midbrain.

<details><summary><weak>Guide</weak></summary>
:-:-: A video guide to completing this project can be viewed <a href="https://vimeo.com/1033609727" target="_blank" rel="noopener noreferrer">here</a>.
</details><hr>


**TASK**: Mount and power your Arduino-based hindbrain (connect the mini-USB cable)
<details><summary><strong>Target</strong></summary>
    The built-in LED on the board should be blinking at 1 Hz.
</details><hr>


**TASK**: Download and install the Arduino IDE (integrated development environment).
- Follow the instructions for your "host" computer's operating system here: [Arduino IDE](https://www.arduino.cc/en/software)
- Open the "Blink" Example: File -> Examples -> Basic -> Blink
- Upload this example to your board
- ***IMPORTANT***: If you have trouble connecting to the Arduino from your Laptop, then it may be necessary to install the "latest" driver from FTDI for the chip that communicates over the USB cable. This is not always necessary, so please try the normal installation first. However, if you are stuck, then please checkout these [FTDI driver installation instructions](https://support.arduino.cc/hc/en-us/articles/4411305694610-Install-or-update-FTDI-drivers).
<details><summary><strong>Target</strong></summary>
    You should be able to successfully compile and upload the "Blink" example (with no errors).
</details><hr>


### NB3 : Building a Theremin
> Building a light-to-sound feedback loop musical instrument (theremin) using an Arduino, an LDR, and a Piezo buzzer.

<details><summary><weak>Guide</weak></summary>
:-:-: A video guide to completing this project can be viewed <a href="https://vimeo.com/1033896646" target="_blank" rel="noopener noreferrer">here</a>.
</details><hr>


**TASK**: Build a Theremin
- *Hint*: What if you used the analog voltage signal measured from your light sensor to change the frequency of the "tone" playing on your buzzer? Hmm...
<details><summary><strong>Target</strong></summary>
    You should here a sound that varies with your hand motion (in front of a light)
</details><hr>


**TASK**: ***Have fun!*** (Make something cool)
- This diagram of the Arduino "pins" will definitely be useful: ![Arduino Pinout](/boxes/computers/_resources/images/pinout_arduino_nano.png)
<details><summary><strong>Target</strong></summary>
    You should have fun!
</details><hr>


### NB3 : Programming Arduino
> An introduction to programming an Arduino microcontroller.

<details><summary><weak>Guide</weak></summary>
:-:-: A video guide to completing this project can be viewed <a href="https://vimeo.com/1033810807" target="_blank" rel="noopener noreferrer">here</a>.
</details><hr>


**TASK**: Blink the internal LED faster (or slower)
- Change the following example code (the classic "[Blinky](/boxes/computers/arduino/ide/blink/blink.ino)" example) to make the LED blink at a different frequency.
```c
/*
  Blink

  Turns an LED on for one second, then off for one second, forever.

  Most Arduinos have an on-board LED that you can control. On the NANO
  (your NB3's Hindbrain) it is attached to pin 13.
*/

// The setup function runs once after you press reset or power up the board
void setup() {
  // Initialize digital pin 13 (the LED) as an output.
  pinMode(13, OUTPUT);
}

// The loop function runs over and over again...forever
void loop() {
  digitalWrite(13, HIGH);   // Turn the LED on (set to the HIGH voltage level)
  delay(1000);              // Wait for a second
  digitalWrite(13, LOW);    // Turn the LED off (set to the LOW voltage level)
  delay(1000);              // Wait for a second
}
```

<details><summary><strong>Target</strong></summary>
    Your Arduino's built-in LED should now be blinking faster or slower than 1 Hz.
</details><hr>


**TASK**: Blink an *external* LED
- *Hint*: Connect the LED to a digital output pin (D13 in the example below), but don't forget your current limiting resistor!
<p align="center">
<img src="../../boxes/computers/_resources/images/LED_driver_circuit.png" alt="LED Driver" width="400">
</p>

<details><summary><strong>Target</strong></summary>
    Your external LED should now be blinking at the same time as the built-in LED (both are connected to pin 13).
</details><hr>


**TASK**: Generate a *pulsing* signal for your piezo buzzer
- This is a piezo buzzer:
<p align="center">
<img src="../../boxes/computers/_resources/images/piezo_buzzer.png" alt="Piezo Buzzer" width="300">
</p>

- The piezo buzzer will expand (5V) and contract (0V) as you switch the voltage applied across it. This expansion/contraction forces air into and out of the plastic case. If you switch it ON/OFF fast enough, then you can *hear it*!
- Connect one leg of the piezo to pin 13 and the other to Ground.
- You could use the "Blink" example...but with a much shorter delay between the ON/OFF "blinks". However, it is much easier to use a function (command) called "tone()" which will allow you generate pulses at specific frequencies.
- Upload this [code](/boxes/computers/arduino/ide/tone/tone.ino) to your Arduino.
```c
/*
  Tone
   - Generate a tone (square wave pulses at a specific frequency) on one of
     Arduino's digital pins
   -- You can use any of the Arduino pins that have "~" symbol on the pinout
      diagram. This example uses pin 11 (PIEZO_PIN)
*/

// List constant values that you can use throughout the program
const int PIEZO_PIN = 11;       // Buzzer Pin (must have ~ for PWM)

// The setup function runs once after you press reset or power up the board
void setup() {
  // Initialize digital pin PIEZO_PIN as an output.
  pinMode(PIEZO_PIN, OUTPUT);
}

// The loop function runs over and over again...forever
void loop() {
  // Generate Sound Output at 2000 Hz (2 kHz) for 1000 ms (1 second)
  tone(PIEZO_PIN, 2000, 1000);

  // Wait 1500 ms for the tone to finish
  // - The tone will play for 1000 ms and then silence for 500 ms
  delay(1500);

  // Generate Sound Output at 1500 Hz (1.5 kHz) for 1000 ms (1 second)
  tone(PIEZO_PIN, 2000, 1000);

  // Wait 1500 ms for the tone to finish
  // - The tone will play for 1000 ms and then silence for 500 ms
  delay(1500);
}
```

- **Challenge**: Try to add some more notes to play a recognizable melody!
<details><summary><strong>Target</strong></summary>
    You should here a (somewhat unpleasant) sound from the piezo buzzer
</details><hr>


**TASK**: Measure an analog signal from your LDR light sensor circuit
- *Hint*: Send the output voltage of your light sensor (the "middle" of the divider) to an analog input pin.
- *Help*: Check out the example in (*File->Examples->Basic->AnalogReadSerial*) to see how to use the "Serial Monitor" to report the analog voltage signal measured from your light sensor back to your host computer.
- *Challenge*: Write a program that will turn on your LED when the light signal is above (or below) some threshold.
<details><summary><strong>Target</strong></summary>
    You should see values on your host laptop
</details><hr>


