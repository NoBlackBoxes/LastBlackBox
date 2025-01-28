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
