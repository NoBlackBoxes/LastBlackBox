/*
  H-Bridge Test
*/

// Globals
const int PWM_PIN = 9;
const int A_PIN = 2;
const int B_PIN = 3;

void setup() {

  // LED Feedback
  pinMode(LED_BUILTIN, OUTPUT);

  // Motor Digital Outputs
  pinMode(PWM_PIN, OUTPUT);  // PWM
  pinMode(A_PIN, OUTPUT);    // A
  pinMode(B_PIN, OUTPUT);    // B

  // Initialize
  digitalWrite(PWM_PIN, LOW);
  digitalWrite(A_PIN, LOW);
  digitalWrite(B_PIN, LOW);
}

void loop() {
  
  // Forward
  forward();
  delay(500);

  // Backward
  backward();
  delay(500);

  // Stop
  stop();
  delay(500);

  // Wait a bit
  delay(10);
}

// Go
void go()
{
  digitalWrite(LED_BUILTIN, HIGH);
  digitalWrite(PWM_PIN, HIGH);
}

// Stop
void stop()
{
  digitalWrite(LED_BUILTIN, LOW);
  digitalWrite(PWM_PIN, LOW);
  digitalWrite(A_PIN, LOW);
  digitalWrite(B_PIN, LOW);
}

// Forward
void forward()
{
  digitalWrite(A_PIN, HIGH);
  digitalWrite(B_PIN, LOW);

  // Move
  go();
}

// Backward
void backward()
{
  digitalWrite(A_PIN, LOW);
  digitalWrite(B_PIN, HIGH);

  // Move
  go();
}
