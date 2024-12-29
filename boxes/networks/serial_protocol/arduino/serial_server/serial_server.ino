/*
  Serial Server
     - Respond to single character commands received via serial
*/

void setup() {
  // Initialize serial port
  Serial.begin(115200);

  // Initialize output pins
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  
  // Check for any incoming bytes
  if (Serial.available() > 0) {
    char new_char = Serial.read();

    // Respond to command "x"
    if(new_char == 'x') {
      // Turn off LED pin 13
      digitalWrite(LED_BUILTIN, LOW);
    }

    // Respond to command "o"
    if(new_char == 'o') {
      // Turn on LED pin 13
      digitalWrite(LED_BUILTIN, HIGH);
    }

  }

  // Wait a bit
  delay(10);
}