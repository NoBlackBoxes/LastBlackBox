# Resources for Building a "Remote-Control" NB3

## Install VS Code on your Chromebook
Open a (Linux) terminal and run the following command
```bash
dpkg --print-architecture
```
It will print the name of your Chromebook's CPU architecture

If your CPU architecture is **x86** or **amd64** or **x64**, then run the following commands:
```bash
# Download VS Code (x64 version)
wget "https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-x64" -O "code-download.deb"

# Install Gnome Keyring
sudo apt-get install gnome-keyring

# Install VS code
sudo apt install ./code-download.deb
# - Follow the on-screen instructions
```

If your CPU architecture is **arm64** or **aarch64**, then run the following commands:
```bash
# Download VS Code (arm64 version)
wget "https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-arm64" -O "code-download.deb"

# Install Gnome Keyring
sudo apt-get install gnome-keyring

# Install VS code
sudo apt install ./code-download.deb
# - Follow the on-screen instructions
```

If your CPU architecture is **arm32** or **aarch32** or **armhf**, then run the following commands:
```bash
# Download VS Code (armhf version)
wget "https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-armhf" -O "code-download.deb"

# Install Gnome Keyring
sudo apt-get install gnome-keyring

# Install VS code
sudo apt install ./code-download.deb
# - Follow the on-screen instructions
```

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
