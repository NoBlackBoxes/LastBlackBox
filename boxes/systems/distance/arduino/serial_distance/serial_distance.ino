#define TRIG_PIN 5  // Trigger pin (rG)
#define ECHO_PIN 6  // Echo pin (lG)

void setup() {
    Serial.begin(115200);       // Start serial communication
    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);
}

void loop() {
    long duration;
    float distance;

    // Send a 10-microsecond pulse to trigger the sensor
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);

    // Measure the pulse duration on the echo pin, 50 ms timeout (~8 m max distance)
    duration = pulseIn(ECHO_PIN, HIGH, 50000);

    // Convert time (microseconds) to distance (millimeters)
    // - Speed of Sound: 343 m/s (there and back = x2)
    distance = duration * 0.343 / 2;

    // Print the result
    Serial.print("Distance: ");
    Serial.print(distance);
    Serial.println(" mm");

    delay(50); // Wait before the next measurement
}