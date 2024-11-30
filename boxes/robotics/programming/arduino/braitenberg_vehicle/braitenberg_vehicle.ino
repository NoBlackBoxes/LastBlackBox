/*
  Braitenberg Vehicle #1 (simple sensory motor loop)
*/

#include <Servo.h>  // This includes the "servo" library

Servo left, right;  // This creates to servo objects, one for each motor

// Setup
void setup() {
  right.attach(9);  // Assign right servo to digital (PWM) pin 9 (change according to your connection)
  left.attach(10);  // Assign left servo to digital (PWM) pin 10 (change according to your connection)
}

void loop() {

  // Sensory Input
  int light_right = analogRead(A0);
  int light_left = analogRead(A1);

  // Which side is brighter?
  if (light_left > light_right)
  {
    // If left is above threshold, then turn to the left
    if(light_left > 750)
    {
      // Motor Output
      left.write(0);      // left backwards
      right.write(0);     // right forwards      
    }
    else
    {
      left.write(90);     // left stop
      right.write(90);    // right stop            
    }
  }
  else
  {
    // If right is above threshold, then turn to the right
    if(light_right > 750)
    {
      // Motor Output
      left.write(180);    // left forwards
      right.write(180);   // right backwards
    }
    else
    {
      left.write(90);     // left stop
      right.write(90);    // right stop            
    }
  }
}
