# Remote-NB3 : Arduino Code

## Upload the "Serial Controller" to your Arduino
This code will run on your Arduino. It listens for single letter commands sent via the serial (USB) cable and responds by moving the wheels.
- Command 'f': Forwards
- Command 'b': Backwards
- Command 'l': Turn left
- Command 'r': Turn right
- Command 'x': Stop

**Note**: Your servo motors must be connected to pin 9 (right servo) and pin 10 (left servo) of your Arduino Nano.

```c
/*
  Serial Controller
    - Respond to single character commands received via serial with servo motion
*/

#include <Servo.h>    // This includes the "servo" library
Servo left, right;    // This creates two servo objects, one for each motor

void setup() {
  // Initialize serial port
  Serial.begin(115200);

  // Attach servo pins
  right.attach(9);    // Assign right servo to digital (PWM) pin 9 (change according to your connection)
  left.attach(10);    // Assign left servo to digital (PWM) pin 10 (change according to your connection)

  // Initialize (no motion)
  left.write(90);
  right.write(90);
}

void loop() {
  
  // Check for any incoming bytes
  if (Serial.available() > 0) {
    char new_char = Serial.read();

    // Respond to command "f"
    if(new_char == 'f') {
      forward();
    }

    // Respond to command "b"
    if(new_char == 'b') {
      backward();
    }

    // Respond to command "l"
    if(new_char == 'l') {
      turn_left();
    }

    // Respond to command "r"
    if(new_char == 'r') {
      turn_right();
    }

    // Respond to command "x"
    if(new_char == 'x') {
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
```

## Remote Control your NB3
If you connect to your NB3's Raspberry Pi via SSH, then you can run commands on your Raspberry Pi remotely. The Python code [here](/boxes/networks/remote-NB3/python/drive/drive.py) listens for a keypress (either the up, down, left, or right arrow) and then sends the appropriate command to your Arduino to driver forward, backward, or turn. You can exit this Python code by pressing 'q' to quit.

To run this code, you can navigate to the folder containing the file and run it with Python.
```bash
cd /home/${USER}/NoBlackBoxes/LastBlackBox/boxes/networks/remote-NB3/python/drive
python drive.py
```

...or you can use a "shortcut" command we have added to your Linux terminal (which just does the same as the above...more succinctly)
```bash
Drive
```
