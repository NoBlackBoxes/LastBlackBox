/*
  Serial Controller
*/

// Globals
int pin_left_PWM = 9;
int pin_left_A = 2;
int pin_left_B = 3;
int pin_right_PWM = 10;
int pin_right_A = 4;
int pin_right_B = 5;

void setup() {
  // Initialize serial port
  Serial.begin(19200);

  // LED Feedback
  pinMode(LED_BUILTIN, OUTPUT);

  // Left Motor Digital Outputs
  pinMode(pin_left_PWM, OUTPUT);  // PWM
  pinMode(pin_left_A, OUTPUT);    // A
  pinMode(pin_left_B, OUTPUT);    // B

  // Right Motor Digital Outputs
  pinMode(pin_right_PWM, OUTPUT);  // PWM
  pinMode(pin_right_A, OUTPUT);    // A
  pinMode(pin_right_B, OUTPUT);    // B

  // Initialize
  digitalWrite(pin_left_PWM, LOW);
  digitalWrite(pin_left_A, LOW);
  digitalWrite(pin_left_B, LOW);
  digitalWrite(pin_right_PWM, LOW);
  digitalWrite(pin_right_A, LOW);
  digitalWrite(pin_right_B, LOW);
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

    // Respond to command "x"
    if(newChar == 'x') {
      stop();
    }
  }

  // Wait a bit
  delay(10);
}

// Go
void go()
{
  digitalWrite(LED_BUILTIN, HIGH);
  digitalWrite(pin_left_PWM, HIGH);
  digitalWrite(pin_right_PWM, HIGH);
}

// Stop
void stop()
{
  digitalWrite(LED_BUILTIN, LOW);
  digitalWrite(pin_left_PWM, LOW);
  digitalWrite(pin_left_A, LOW);
  digitalWrite(pin_left_B, LOW);
  digitalWrite(pin_right_PWM, LOW);
  digitalWrite(pin_right_A, LOW);
  digitalWrite(pin_right_B, LOW);
}

// Forward
void forward()
{
  // Left motor
  digitalWrite(pin_left_A, HIGH);
  digitalWrite(pin_left_B, LOW);
  
  // Right motor
  digitalWrite(pin_right_A, HIGH);
  digitalWrite(pin_right_B, LOW);

  // Move
  go();
}

// Backward
void backward()
{
  // Left motor
  digitalWrite(pin_left_A, LOW);
  digitalWrite(pin_left_B, HIGH);
  
  // Right motor
  digitalWrite(pin_right_A, LOW);
  digitalWrite(pin_right_B, HIGH);

  // Move
  go();
}

// Left
void left()
{
  // Left motor
  digitalWrite(pin_left_A, LOW);
  digitalWrite(pin_left_B, HIGH);
  
  // Right motor
  digitalWrite(pin_right_A, HIGH);
  digitalWrite(pin_right_B, LOW);

  // Move
  go();
}

// Right
void right()
{
  // Left motor
  digitalWrite(pin_left_A, HIGH);
  digitalWrite(pin_left_B, LOW);
  
  // Right motor
  digitalWrite(pin_right_A, LOW);
  digitalWrite(pin_right_B, HIGH);

  // Move
  go();
}
