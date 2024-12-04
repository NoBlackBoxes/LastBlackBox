/*
  Blink and Sing (concurrent version)
*/

// Includes
#include "defs.h"               // This includes a local header with constant definitions

// Constants
const int LED_PIN = 13;         // LED Pin
const int PIEZO_PIN = 11;       // Pizeo Buzzer Pin

// Setup (run once at start)
void setup() {
  pinMode(LED_PIN, OUTPUT);
  pinMode(PIEZO_PIN, OUTPUT);
}

// Loop (run over and over again)
void loop() {

  // Get current time (in milliseconds)
  int current_time = millis();

  // Schedule actions
  blink(current_time);
  sing(current_time);
}

// Blink Function
void blink(int current_time) {

  // Static variables
  static bool state = false;
  static int previous_time = 0;
  static int next_interval = 100;
  
  // Measure elapsed time
  int elapsed_time = current_time - previous_time;

  // Do something?
  if (elapsed_time > next_interval)
  {
    // Change LED state
    state = !state;
    digitalWrite(LED_PIN, state);
    
    // Set next interval
    next_interval = 100;    

    // Reset previous time
    previous_time = current_time;    
  }

  return;
}

// Sing Function
#define NUM_NOTES 98
int sing(int current_time){
  
  // Static variables
  static int counter = 0;
  static int previous_time = 0;
  static int next_interval = 100;

  // Song arrays
  int notes[NUM_NOTES] =     {G4,E5,D5,C5,G4,G4,E5,D5,C5,A4,A4,F5,E5,D5,B4,G5,G5,F5,D5,E5,G4,E5,D5,C5,G4,G4,E5,D5,C5,A4,A4,F5,E5,D5,G5,G5,G5,G5,G5,A5,G5,F5,D5,C5,G5,E5,E5,E5,E5,E5,E5,E5,G5,C5,D5,E5,F5,F5,F5,F5,F5,F5,E5,E5,E5,E5,E5,D5,D5,E5,D5,G5,E5,E5,E5,E5,E5,E5,E5,G5,C5,D5,E5,F5,F5,F5,F5,F5,F5,E5,E5,E5,E5,G5,G5,F5,D5,C5};
  int durations[NUM_NOTES] = {8,8,8,8,2,8,8,8,8,2,8,8,8,8,2,8,8,8,8,2,8,8,8,8,2,8,8,8,8,2,8,8,8,8,8,8,8,16,16,8,8,8,8,4,4,8,8,4,8,8,4,8,8,8,8,2,8,8,8,16,16,8,8,8,16,16,8,8,8,8,4,4,8,8,4,8,8,4,8,8,8,8,2,8,8,8,16,16,8,8,8,16,16,8,8,8,8,2};

  // Measure elapsed time
  int elapsed_time = current_time - previous_time;

  // Do something?
  if (elapsed_time > next_interval)
  {
    // Set next interval
    int tempo = 2;
    next_interval = (10000/durations[counter])/tempo;
    
    // Play next note
    tone(PIEZO_PIN, notes[counter], next_interval - 25);

    // Increment note counter
    counter = (counter + 1) % NUM_NOTES;

    // Reset previous time
    previous_time = current_time;    
  }
  
  return;  
}
