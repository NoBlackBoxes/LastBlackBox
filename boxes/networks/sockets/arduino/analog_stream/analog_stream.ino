// Stream analog A0 values as uint8 bytes
void setup() {
  Serial.begin(115200);
}

void loop() {
  int value = analogRead(A0);           // Measure (0–1023)
  uint8_t scaled = value >> 2;          // Convert to 0–255
  Serial.write(scaled);                 // Send one raw byte
  delayMicroseconds(1000);              // Limit sample rate
}
