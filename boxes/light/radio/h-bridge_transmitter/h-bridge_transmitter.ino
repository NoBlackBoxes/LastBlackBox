// Pin definitions
const int ENABLE_PIN = 9;
const int pin1 = 10;
const int pin2 = 11;
const int DOT_LENGTH = 30000;           // Dot length (2 x microseconds)
const int DASH_LENGTH = 90000;          // Dash length (2 x microseconds)
const int INTER_SYMBOL_LENGTH = 100;    // Symbol interval length (milliseconds)
const int INTER_LETTER_LENGTH = 200;    // Letter interval length (milliseconds)
const int INTER_WORD_LENGTH = 400;      // Word interval length (milliseconds)

// Delay in microseconds
const unsigned int delay_us = 1;

void setup() {
  // Initialize pins as outputs
  pinMode(ENABLE_PIN, OUTPUT);
  pinMode(pin1, OUTPUT);
  pinMode(pin2, OUTPUT);
}

void loop() {
  dash(); dot(); pause();                 // N
  dash(); dash(); dash();                 // O
  space();
  dash(); dot(); dot(); dot(); pause();   // B
  dot(); dash(); dot(); dot(); pause();   // L
  dot(); dash(); pause();                 // A
  dash(); dot(); dash(); dot(); pause();  // C
  dash(); dot(); dash(); pause();         // K
  space();
  dash(); dot(); dot(); dot(); pause();   // B  
  dash(); dash(); dash();                 // O
  dash(); dot(); dot(); dash(); pause();  // X
  dot(); pause();                         // E
  dot(); dot(); dot(); pause();           // S  
  delay(1000);
}

void dot()
{
  int counter = 0;
  digitalWrite(ENABLE_PIN, HIGH);
  for(counter = 0; counter < DOT_LENGTH; counter++)
  {
    // Toggle Left
    digitalWrite(pin1, HIGH);
    digitalWrite(pin2, LOW);
    delayMicroseconds(delay_us);
  
    // Toggle Right
    digitalWrite(pin1, LOW);
    digitalWrite(pin2, HIGH);
  }
  digitalWrite(ENABLE_PIN, LOW);
  delay(INTER_SYMBOL_LENGTH);
}


void dash()
{
  int counter = 0;
  digitalWrite(ENABLE_PIN, HIGH);
  for(counter = 0; counter < DASH_LENGTH; counter++)
  {
    // Toggle Left
    digitalWrite(pin1, HIGH);
    digitalWrite(pin2, LOW);
    delayMicroseconds(delay_us);
  
    // Toggle Right
    digitalWrite(pin1, LOW);
    digitalWrite(pin2, HIGH);
  }
  digitalWrite(ENABLE_PIN, LOW);
  delay(INTER_SYMBOL_LENGTH);
}

void pause()
{
  delay(INTER_LETTER_LENGTH);
}
void space()
{
  delay(INTER_WORD_LENGTH);
}