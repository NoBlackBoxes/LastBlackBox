#include "Arduino.h"

// Pin definitions
const int ANT_PIN = 9;                  // Antenna Pin
const int LED_PIN = 13;                 // LED Pin
const int DOT_LENGTH = 100;             // Dot length (milliseconds)
const int DASH_LENGTH = 300;            // Dash length (milliseconds)
const int INTER_SYMBOL_LENGTH = 100;    // Symbol interval length (milliseconds)
const int INTER_LETTER_LENGTH = 200;    // Letter interval length (milliseconds)
const int INTER_WORD_LENGTH = 400;      // Word interval length (milliseconds)

/*
Frequency Table

The OCR1A variable is one less than the actual divisor.
OCR1A - Frequency

15 - 500 Khz
14 - 530 Khz
13 - 570 Khz
12 - 610 Khz
11 - 670 Khz
10 - 730 Khz
9 -  800 Khz
8 -  890 Khz
7 -  1000 Khz
6 -  1140 Khz
5 -  1330 Khz
4 -  1600 Khz
*/

void setup() {
  // Setup internal timer on ANT_PIN
  TCCR1A = _BV(COM1A0);             // Toggle OC1A (ANT_PIN) on compare match
  TCCR1B = _BV(WGM12) | _BV(CS10);  // CTC, no prescaler
  OCR1A = 11;                       // Set frequency from table

  // Set LED_PIN to output
  pinMode(LED_PIN, OUTPUT);
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
  digitalWrite(LED_PIN, HIGH);
  for(counter = 0; counter < DOT_LENGTH; counter++)
  {
    // Toggle at audible frequency
    pinMode(ANT_PIN, OUTPUT);
    delayMicroseconds(500);
    pinMode(ANT_PIN, INPUT); 
    delayMicroseconds(500);
  }
  digitalWrite(LED_PIN, LOW);
  delay(INTER_SYMBOL_LENGTH);
}


void dash()
{
  int counter = 0;
  digitalWrite(LED_PIN, HIGH);
  for(counter = 0; counter < DASH_LENGTH; counter++)
  {
    // Toggle at audible frequency
    pinMode(ANT_PIN, OUTPUT);
    delayMicroseconds(500);
    pinMode(ANT_PIN, INPUT); 
    delayMicroseconds(500);
  }
  digitalWrite(LED_PIN, LOW);
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