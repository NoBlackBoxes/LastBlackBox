# Computers : NB3: Programming Arduino
An introduction to programming an Arduino microcontroller.

## [Video](https://vimeo.com/1033810807)

## Concepts
- Developing a simple project with Arduino

## Connections

## Lesson
- **TASK**: Blink the internal LED faster (or slower)
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
> Your Arduino's built-in LED should now be blinking faster or slower than 1 Hz.

- **TASK**: Blink an *external* LED 
- *Hint*: Connect an LED to digital pin 13, but don't forget your current limiting resistor!
- ![LED Driver:400](/boxes/computers/_resources/images/LED_driver_circuit.png)
> Your external LED should now be blinking at the same time as the built-in LED (both are connected to pin 13).

- **TASK**: Generate a *pulsing* signal for your piezo buzzer
- This is a piezo buzzer:
- ![Piezo Buzzer:300](/boxes/computers/_resources/images/piezo_buzzer.png)
- The piezo buzzer will expand (5V) and contract (0V) as you switch the voltage applied across it. This expansion/contraction forces air into and out of the plastic case. If you switch it ON/OFF fast enough, then you can *hear it*!
- Connect one leg of the piezo to pin 11 and the other to Ground.
- You could use the "Blink" example...but with a much shorter delay between the ON/OFF "blinks". However, it is much easier to use a function (command) called "tone()" which will allow you generate pulses at specific frequencies.
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

  // Generate Sound Output at 1500 Hz (1.5 kHz) for 1000 ms (1 second)
  tone(PIEZO_PIN, 2000, 1000);

  // Wait 1500 ms for the tone to finish 
  // - The tone will play for 1000 ms and then silence for 500 ms
  delay(1500);
}
```
- **Challenge**: Try to add some more notes to play a recognizable melody!
> You should here a (somewhat unpleasant) sound from the piezo buzzer

- **TASK**: Measure an analog signal from your LDR light sensor circuit
- *Hint*: Send the output voltage of your light sensor (the "middle" of the divider) to an analog input pin.
- *Help*: Check out the example in (*File->Examples->Basic->AnalogReadSerial*) to see how to use the "Serial Monitor" to report the analog voltage signal measured from your light sensor back to your host computer.
- *Challenge*: Write a program that will turn on your LED when the light signal is above (or below) some threshold.
> You should see values on your host laptop

