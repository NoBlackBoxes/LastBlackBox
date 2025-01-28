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