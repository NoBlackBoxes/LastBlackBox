/*
  Muscles Test (Servos)

  Servos are "easier" to control with a computer, but they hide what is going on.

   - In the beginning, a number of "hobby" motor manufacturers got together ~40 years ago to agree on a "standard" 
   to make it easier to produce remote control cars and airplanes. Sending digital pulses was easy (even over a radio), 
   but sending "analog" values was prone to noise and required more expensive transmitters/receivers. They decided that 
   a 1 ms pulse would be "0", a 1.5 ms pulse would be "90", and a 2 ms pulse would be "180". Therefore, a position servo 
   (that rotates to a specific angle) just needs to detect the length of pulses it is getting (between 1 and 2 ms) and it 
   knows exactly what angle to rotate to. Your servos use this same convention, but these pulses set the speed, 1 ms 
   (full backwards), 1.5 ms (stop), 2 ms (full forwards). The chip inside the servo PCB has circuits that measure the duration 
   of incoming pulses (on the orange wire) and then generate the commands to a PWM to control motor speed and an H-bridge 
   to set direction. This chip obeys the "hobby motor" standard. The "servo library" in Arduino is just a convenient way of 
   generating 1 to 2 ms pulses on specific digital pins. You could generate them however you like (using a for loop with a delay 
   (in microseconds). In fact, the Arduino "servo library" interprets any value you send that  is greater than 200 (I think) 
   as a pulse length in microseconds...so sending 1500 should be the same as sending 90. Worth testing...
   but definitely worth knowing what is going on.
*/

#include <Servo.h>  // This includes the "servo" library

Servo left, right;  // This creates to servo objects, one for each motor

int speed = 0;      // This creates a variable called "speed" that is initially set to 0

// Setup
void setup() {
  right.attach(9);  // Assign right servo to digital (PWM) pin 9 (change according to your connection)
  left.attach(10);  // Assign left servo to digital (PWM) pin 10 (change according to your connection)
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