void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);

  // init LED pin
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  // Here you can now write different strategies how to handle and react to serial commands
   if (Serial.available() > 0) {
    char newChar = Serial.read();
    
    // e.g. if the character that was send is "o", turn LED on
    if(newChar == 'o') {
      digitalWrite(LED_BUILTIN, HIGH);
    }

    // if there's something else sent, turn it off
    else{
      digitalWrite(LED_BUILTIN, LOW);
    }
    delay(10);
   }
}
