# A voice controlled NB3

Based on this project: https://github.com/google-coral/project-keyword-spotter

1. Install your NB3 Ears and make sure they can record audio

2. Upload this code to your Arduino, making the correct changes to the pin numbers for your H-bridge

```txt
/*
  Voice Behaviours
*/

void setup() {
  // Initialize serial port
  Serial.begin(19200);

  // LED Feedback
  pinMode(LED_BUILTIN, OUTPUT);

  // Right Motor Digital Outputs
  pinMode(3, OUTPUT); // PWM
  pinMode(4, OUTPUT); // A
  pinMode(5, OUTPUT); // B

  // Left Motor Digital Outputs
  pinMode(6, OUTPUT); // PWM
  pinMode(7, OUTPUT); // A
  pinMode(8, OUTPUT); // B

  // Initialize
  digitalWrite(3, LOW);
  digitalWrite(4, LOW);
  digitalWrite(5, LOW);
  digitalWrite(6, LOW);
  digitalWrite(7, LOW);
  digitalWrite(8, LOW);
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
      left();
    }

    // Respond to command "r"
    if(newChar == 'r') {
      right();
    }
  }

  // Wait a bit
  delay(10);
}

// Go
void go()
{
  digitalWrite(LED_BUILTIN, HIGH);
  digitalWrite(3, HIGH);
  digitalWrite(6, HIGH);
}

// Stop
void stop()
{
  digitalWrite(LED_BUILTIN, LOW);
  digitalWrite(3, LOW);
  digitalWrite(4, LOW);
  digitalWrite(5, LOW);  
  digitalWrite(6, LOW);
  digitalWrite(7, LOW);
  digitalWrite(8, LOW);
}

// Forward
void forward()
{
  // Left motor
  digitalWrite(4, HIGH);
  digitalWrite(5, LOW);
  
  // Right motor
  digitalWrite(7, HIGH);
  digitalWrite(8, LOW);

  // Move
  go();
  delay(1000);
  stop();
}

// Backward
void backward()
{
  // Left motor
  digitalWrite(4, LOW);
  digitalWrite(5, HIGH);
  
  // Right motor
  digitalWrite(7, LOW);
  digitalWrite(8, HIGH);

  // Move
  go();
  delay(1000);
  stop();
}

// Left
void left()
{
  // Left motor
  digitalWrite(4, HIGH);
  digitalWrite(5, LOW);
  
  // Right motor
  digitalWrite(7, LOW);
  digitalWrite(8, HIGH);

  // Move
  go();
  delay(1000);
  stop();
}

// Right
void right()
{
  // Left motor
  digitalWrite(4, LOW);
  digitalWrite(5, HIGH);
  
  // Right motor
  digitalWrite(7, HIGH);
  digitalWrite(8, LOW);

  // Move
  go();
  delay(1000);
  stop();
}
```

3. On RPi (NB3)

```bash
git clone https://github.com/google-coral/project-keyword-spotter

cd project-keyword-spotter

# Copy the run_robot.py file to this folder (the repo's root folder)

# Run robot!
python3 run_robot.py
```