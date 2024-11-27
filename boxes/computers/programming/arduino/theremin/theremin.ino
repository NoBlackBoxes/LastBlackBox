/*
  Theremin
   - A light-to-sound feedback loop musical instrument (theremin) using an LDR and a Piezo buzzer.
*/

const int PIEZO_PIN = 11;       // Buzzer Pin (must have ~/PWM)
const int LDR_PIN = A0;         // Light Sensor Pin (analog)

// The setup function runs once when you press reset or power the board
void setup() {
  // Initialize digital pin PIEZO_PIN as an output.
  pinMode(PIEZO_PIN, OUTPUT);

  // Setup Serial Communication
  Serial.begin(9600);           // Speak serial at 9600 bits per second
}

// The loop function runs over and over again forever
void loop() {
  // Measure Analog Input
  int light = analogRead(LDR_PIN);

  // Send light value (print) to the serial port
  Serial.println(light);

  // Wait a bit (10 ms)
  delay(10);

  // Generate Sound Output
  int sound = 10 * light;       // Sound frequency is 10 times the measured light value
  tone(PIEZO_PIN, sound);
}
