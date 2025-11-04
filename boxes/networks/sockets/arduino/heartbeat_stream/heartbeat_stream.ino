// Heartbeat: Stream analog A0 values as uint8 bytes, while blinking the built-in LED (D13)
//  - Connect D13 to A0

// Heartbeat parameters
int pulse_rate = 2;                       // Specify pulse rate (Hz)
int count_per_pulse = 1000 / pulse_rate;  // Number of counts (1 ms ticks) per pulse transition
int counter = 0;                          // Tick counter
bool pulse_state = false;                 // Pulse state (high/low)

void setup() {
  Serial.begin(115200);                   // Start serial port
  pinMode(13, OUTPUT);                    // Initialize digital pin 13 (the LED) as an output.
}

void loop() {

  // Analog Stream
  int value = analogRead(A0);             // Measure (0–1023)
  uint8_t scaled = value >> 2;            // Convert to 0–255
  Serial.write(scaled);                   // Send one raw byte
  delayMicroseconds(1000);                // Limit sample rate (~1 ms per sample)

  // Heartbeat
  if (counter < count_per_pulse)
  {
    counter++;                            // Increment counter
  }
  else
  {
    pulse_state = !pulse_state;           // Toggle pulse state
    digitalWrite(13, pulse_state);        // Toggle LED
    counter = 0;                          // Reset counter 
  }
}
