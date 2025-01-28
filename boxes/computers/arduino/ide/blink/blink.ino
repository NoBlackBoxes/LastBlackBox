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