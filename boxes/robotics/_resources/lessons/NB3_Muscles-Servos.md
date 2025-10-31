# Robotics : NB3 : Muscles (Servos)
Let's build your robot's movement system (using servo motors).

## [Video](https://vimeo.com/1034800702)

## Concepts
- Add servos as muscles
- Test servos

## Lesson

- **TASK**: Mount the servo motors and wheels to your NB3.
> The mounted servo motors should look like this.

- In order to control your servo motors, you must send a square wave signal from your NB3's hindbrain with very specific timing. The details of this control signal's timing are described in the comments of the example code here: [Servo Test (Arduino)](/boxes/robotics/programming/arduino/muscles_test_servo/muscles_test_servo.ino).
- This servo test code uses a library, called "servo", to make it easier to control your NB3's muscles.
- *code*
```c
#include <Servo.h>  // This includes the "servo" library

Servo left, right;  // This creates two servo objects, one for each motor

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
```

- **TASK**: Test your servo motors by sending control commands from your NB3's hindbrain.
> One servo motor should spin forwards and backwards, then the other...and then repeat.
