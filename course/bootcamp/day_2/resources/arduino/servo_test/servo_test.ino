/*
  Servo Test
*/
#include <Servo.h>  // This includes the "servo" library

Servo left, right;  // This creates to servo objects, one for each motor

int speed = 0;      // This creates a variable called "speed" that is intially set to 0

// Setup
void setup() {
  right.attach(9);  // Assign right servo to digital (PWM) pin 9 (change accorinding to your connection)
  left.attach(10);  // Assign left servo to digital (PWM) pin 10 (change accorinding to your connection)
}

void loop() {

  // Servos are often used to control "angle" of the motor, therefore the "servo library" uses a range of 0 to 180.
  // Your servos control "speed", therefore 0 is full speed clockwise, 90 is stopped, and 180 is full speed counter-clockwise

  // Move left servo through the full range of speeds
  for (speed = 0; speed <= 180; speed += 1) {
    left.write(speed);
    delay(15);
  }
  left.write(90); // stop left servo
  
  // Move right servo
  for (speed = 0; speed <= 180; speed += 1) {
    right.write(speed);
    delay(15);
  }
  right.write(90); // stop right servo
}