/*
  Serial Controller
    - Respond to single character commands received via serial with servo motion
*/

#include <Servo.h>    // This includes the "servo" library
Servo left, right;    // This creates to servo objects, one for each motor

void setup() {
  // Initialize serial port
  Serial.begin(19200);

  // Attach servo pins
  right.attach(9);    // Assign right servo to digital (PWM) pin 9 (change accorinding to your connection)
  left.attach(10);    // Assign left servo to digital (PWM) pin 10 (change accorinding to your connection)

  // Initialize (no motion)
  left.write(90);
  right.write(90);
}

void loop() {
  
  // Check for any incoming bytes
  if (Serial.available() > 0) {
    char newChar = Serial.read();

    // Respond to command "f"
    if(newChar == 'f') {
      forward();
    }

    // Respond to command "b"
    if(newChar == 'b') {
      backward();
    }

    // Respond to command "l"
    if(newChar == 'l') {
      turn_left();
    }

    // Respond to command "r"
    if(newChar == 'r') {
      turn_right();
    }

    // Respond to command "x"
    if(newChar == 'x') {
      stop();
    }
  }

  // Wait a bit
  delay(10);
}

// Forward
void forward()
{
  left.write(180);
  right.write(0);
}

// Backward
void backward()
{
  left.write(0);
  right.write(180);
}

// Left
void turn_left()
{
  left.write(0);
  right.write(0);
}

// Right
void turn_right()
{
  left.write(180);
  right.write(180);
}

// Stop
void stop()
{
  left.write(90);
  right.write(90);
}
