/*
  Serial Server
     - Respond to single character commands received via serial
*/

void setup() {
  // initialize serial port
  Serial.begin(19200);

  // Initialize output pins
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  
  // Check for any incoming bytes
  if (Serial.available() > 0) {
    char newChar = Serial.read();

    // Respond to command "o"
    if(newChar == 'x') {
      // Turn off LED pin 13
      digitalWrite(LED_BUILTIN, LOW);
    }

    // Respond to command "x"
    if(newChar == 'o') {
      // Turn on LED pin 13
      digitalWrite(LED_BUILTIN, HIGH);
    }

  }

  // Wait a bit
  delay(10);
}