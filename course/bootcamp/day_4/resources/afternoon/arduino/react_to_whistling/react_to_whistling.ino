void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);

  // init LED pin
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
   if (Serial.available() > 0) {
    char newChar = Serial.read();
    
    if(newChar == 'o') {
      digitalWrite(LED_BUILTIN, HIGH);
    }
    else{
      digitalWrite(LED_BUILTIN, LOW);
    }
    delay(10);
   }
}
