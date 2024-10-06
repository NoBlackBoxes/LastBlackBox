/*
  Demo

  Classic Arduino example used to test and demonstrate the NBB "code-explainer"
  and "explainer-recorder VS Code extensions

*/

// The setup function runs once when you press reset or power the board
void setup()
{
    // Initialize digital pin LED_BUILTIN as an output.
    pinMode(LED_BUILTIN, OUTPUT);
}

// The loop function runs over and over again forever
void loop()
{
    digitalWrite(LED_BUILTIN, HIGH); // turn the LED on (HIGH is the voltage level)
    delay(500);                      // wait for a second
    digitalWrite(LED_BUILTIN, LOW);  // turn the LED off by making the voltage LOW
    delay(200);                      // wait for a second
}