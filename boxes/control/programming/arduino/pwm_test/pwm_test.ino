/*
 * PWM Test
 *  - Generate a PWM digital signal that ramps down from max (255) to min (0) duty cycle, then back up.
 */

const int PWM_PIN = 11;

void setup() {
  pinMode(OUTPUT, PWM_PIN);
}

void loop() {
  int i;
  
  // Generate 100% to 0% ramp in duty cycle (PWM)
  for(i = 255; i >= 0; i--)
  {
    analogWrite(PWM_PIN, i);
    delay(5);
  }
  delay(1000);
  
  // Generate 0% to 100% ramp in duty cycle (PWM)
  for(i = 0; i <= 255; i++)
  {
    analogWrite(PWM_PIN, i);
    delay(5);
  }
  delay(1000);
}
