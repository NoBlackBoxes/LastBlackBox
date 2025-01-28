# Computers : NB3: Programming Arduino
An introduction to programming an Arduino microcontroller.

## [Video](https://vimeo.com/1033810807)

## Concepts
- Developing a simple project with Arduino

## Connections

## Lesson
- You will now write programs that interact with the input and output pins of your Arduino. This "pin diagram" will help you find the correct locations. ***The Arduino on your NB3 is mounted upside down relative to this diagram...adjust accordingly!*** 
- ![Arduino Nano Pin Diagram:500](/boxes/computers/_resources/images/pinout_arduino_nano.png)

- **TASK**: Blink an *external* LED 
- Change the following example code (the classic "[Blinky](/boxes/computers/arduino/ide/blink/blink.ino)" example) to make the LED blink at a different frequency.
- *code*
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
> Your external LED should now be blinking faster or slower than 1 Hz (once per second).

- **TASK**: Blink an *external* LED 
- *Hint*: Connect an LED to digital pin 13, but don't forget your current limiting resistor!
- ![LED Driver:400](/boxes/computers/_resources/images/LED_driver_circuit.png)
> Your external LED should now be blinking at the same time as the built-in LED (both are connected to pin 13).

- **TASK**: Generate a *pulsing* signal for your piezo buzzer
- This is a piezo buzzer:
- ![Piezo Buzzer:300](/boxes/computers/_resources/images/piezo_buzzer.png)

- The piezo buzzer will expand and contract as you switch the voltage applied across it from 0V to 5V. This expansion and contraction forces air into and out of the plastic case. If you switch it ON/OFF fast enough, then you can *hear it*!
- Connect one leg of the piezo to pin 11 and the other to Ground.
- You could use the "Blink" example to toggle pin 11 with a much shorter delay between the ON/OFF "blinks". However, it is much easier to use a function (command) called "tone()" that will allow you generate pulses at specific frequencies.
- Upload this [code](/boxes/computers/arduino/ide/tone/tone.ino) to your Arduino.
- *code*
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
> You should here a (somewhat unpleasant) sound from the piezo buzzer

- **TASK**: Measure an **analog** signal from your LDR light sensor circuit and send the measured values to your host computer via the USB (serial) connection.
- *Hint*: Connect the output voltage of your light sensor (the "middle" of the divider) to an analog input pin (the example below uses pin A0).
- *Note*: In order to see what values are measured, the following program sends the analog values as text characters over the USB serial connection to your laptop. Your can watch these values arrive by opening the Arduino IDE's "Serial Monitor" (an icon in the upper-right corner of the main window).
- Upload this [code](/boxes/computers/arduino/ide/tone/tone.ino) to your Arduino.
- *code*
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
> You should see values on your host laptop and they should change along with changing light levels.
