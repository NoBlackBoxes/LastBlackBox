# The Last Black Box : Computers
In this box, you will learn about computers...

## Computers
It may not yet seem believable, but you can build a **computer** by combining transistors in a clever way. **Let's learn how!**

<details><summary><i>Materials</i></summary><p>

Name|Depth|Description| # |Data|Link|
:-------|:---:|:----------|:-:|:--:|:--:|
Microcontroller|01|Arduino Nano (rev.3)|1|[-D-](/boxes/computers/_resources/datasheets/arduino_nano_rev3.pdf)|[-L-](https://uk.farnell.com/arduino/a000005/arduino-nano-evaluation-board/dp/1848691)
Piezo Buzzer|01|Piezoelectric speaker/transducer|1|[-D-](/boxes/computers/_resources/datasheets/piezo_buzzer.pdf)|[-L-](https://uk.farnell.com/tdk/ps1240p02bt/piezoelectric-buzzer-4khz-70dba/dp/3267212)
Cable (MiniUSB-1m)|01|Mini-USB to Type-A cable (1 m)|1|[-D-](/boxes/computers/)|[-L-](https://www.amazon.co.uk/gp/product/B07FWF2KBF)

</p></details><hr>

#### Watch this video: 
<p align="center">
<a href="https://vimeo.com/1033601146" title="Control+Click to watch in new tab"><img src="../../boxes/computers/_resources/lessons/thumbnails/Architecture.gif" alt="Architecture" width="480"/></a>
</p>

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

- You will now write programs that interact with the input and output pins of your Arduino. This "pin diagram" will help you find the correct locations. ***The Arduino on your NB3 is mounted upside down relative to this diagram...adjust accordingly!***
<p align="center">
<img src="../../boxes/computers/_resources/images/pinout_arduino_nano.png" alt="Arduino Nano Pin Diagram" width="500">
</p>


**TASK**: Blink an *external* LED
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
    Your external LED should now be blinking faster or slower than 1 Hz (once per second).
</details><hr>


**TASK**: Blink an *external* LED
- *Hint*: Connect an LED to digital pin 13, but don't forget your current limiting resistor!
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

- The piezo buzzer will expand and contract as you switch the voltage applied across it from 0V to 5V. This expansion and contraction forces air into and out of the plastic case. If you switch it ON/OFF fast enough, then you can *hear it*!
- Connect one leg of the piezo to pin 11 and the other to Ground.
- You could use the "Blink" example to toggle pin 11 with a much shorter delay between the ON/OFF "blinks". However, it is much easier to use a function (command) called "tone()" that will allow you generate pulses at specific frequencies.
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

  // Generate Sound Output at 1700 Hz (1.7 kHz) for 1000 ms (1 second)
  tone(PIEZO_PIN, 1700, 1000);

  // Wait 1500 ms for the tone to finish
  // - The tone will play for 1000 ms and then silence for 500 ms
  delay(1500);
}
```

- **Challenge**: Try to add some more notes to play a recognizable melody!
<details><summary><strong>Target</strong></summary>
    You should here a (somewhat unpleasant) sound from the piezo buzzer
</details><hr>


**TASK**: Measure an **analog** signal from your LDR light sensor circuit and send the measured values to your host computer via the USB (serial) connection.
- *Hint*: Connect the output voltage of your light sensor (the "middle" of the divider) to an analog input pin (the example below uses pin A0).
- *Note*: In order to see what values are measured, the following program sends the analog values as text characters over the USB serial connection to your laptop. Your can watch these values arrive by opening the Arduino IDE's "Serial Monitor" (an icon in the upper-right corner of the main window).
- Upload this [code](/boxes/computers/arduino/ide/tone/tone.ino) to your Arduino.
```c
/*
  Analog
  - Reads an analog voltage on pin A0
  - Sends the measured value to the USB serial port.
*/

// The setup function runs once after you press reset or power up the board
void setup() {
  // Initialize serial communication at 9600 bits per second
  Serial.begin(9600);
}

// The loop function runs over and over again...forever
void loop() {
  // Read the input on analog pin A0
  int value = analogRead(A0);

  // Send (print) the value on the serial port
  Serial.println(value);
  delay(1); // Wait briefly (1 ms) between reads for stability
}
```

- *Challenge*: Write a program that will turn on your LED when the light signal is above (or below) some threshold value.
<details><summary><strong>Target</strong></summary>
    You should see values on your host laptop and they should change along with changing light levels.
</details><hr>


