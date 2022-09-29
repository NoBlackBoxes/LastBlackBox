/*
  Analog (PWM) Controller
*/

void setup() {
  // Initialize serial port
  Serial.begin(19200);

  // Right Motor Digital Outputs
  pinMode(3, OUTPUT); // B
  pinMode(4, OUTPUT); // A
  pinMode(5, OUTPUT); // PWM

  // Left Motor Digital Outputs
  pinMode(6, OUTPUT); // PWM
  pinMode(7, OUTPUT); // A
  pinMode(8, OUTPUT); // B

  // Initialize
  digitalWrite(3, LOW);
  digitalWrite(4, LOW);
  analogWrite(5, 0);
  analogWrite(6, 0);
  digitalWrite(7, LOW);
  digitalWrite(8, LOW);
}

// Create message bytes
byte message[4];

void loop() {
  
  // Check for any incoming messages (4 bytes: L direction, L speed (0-255), R direction, R speed (0-255))
  if (Serial.available() >= 4) {

    // Read message
    int num_bytes = Serial.readBytes(message, 4);

    // Parse bytes
    byte l_direction = message[0];
    byte l_speed = message[1];
    byte r_direction = message[2];
    byte r_speed = message[3];

    // Set direction (left)
    if(l_direction == 0)
    {
        digitalWrite(3, LOW);
        digitalWrite(4, HIGH);
    } 
    else
    {
        digitalWrite(4, LOW);
        digitalWrite(3, HIGH);
    }

    // Set direction (right)
    if(r_direction == 0)
    {
        digitalWrite(8, LOW);
        digitalWrite(7, HIGH);
    } 
    else
    {
        digitalWrite(7, LOW);
        digitalWrite(8, HIGH);
    }

    // Set speeds
    analogWrite(5, l_speed);
    analogWrite(6, r_speed);
  }
    
  // Wait a bit
  delay(5);
}
