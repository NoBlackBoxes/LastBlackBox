#include <Arduino.h>
#include <math.h>

const size_t BufferSize = 20;
// Note: if you increase the BufferSize too much, the memory of the Arduino will not be sufficient anymore


class RollingStatistics {
private:
    double buffer[BufferSize];  // Circular buffer to store data points
    size_t currentIndex;        // Current index in the circular buffer
    size_t count;               // Number of valid data points in the buffer
    double sum;                 // Sum of the data points in the buffer

public:
    RollingStatistics() : currentIndex(0), count(0), sum(0) {}

    void addDataPoint(double dataPoint) {
        if (count == BufferSize) {
            sum -= buffer[currentIndex]; // Remove the oldest data point from the sum
        } else {
            count++;
        }

        buffer[currentIndex] = dataPoint; // Add the new data point to the buffer
        sum += dataPoint;                 // Update the sum

        currentIndex = (currentIndex + 1) % BufferSize; // Update the current index
    }

    double getAverage() {
        if (count == 0) {
            return 0.0;
        }

        return sum / count; // Calculate and return the average
    }

    double getStandardDeviation() {
        if (count == 0) {
            return 0.0;
        }

        double mean = sum / count;
        double sumSquares = 0.0;

        for (size_t i = 0; i < count; i++) {
            double deviation = buffer[i] - mean;
            sumSquares += deviation * deviation;
        }

        double variance = sumSquares / count;

        return sqrt(variance); // Calculate and return the standard deviation
    }
};

RollingStatistics rs;


void setup() {

  // put your setup code here, to run once:
  Serial.begin(9600);

  // init light sensor pin
  pinMode(A0, INPUT);
  
  // init LED pin
  pinMode(LED_BUILTIN, OUTPUT);
}



void loop() {

  // read light sensor and print value
  int lightSensorValue = analogRead(A0);
 

  // we added a class that calculates the rolling standard deviation using a fixed window size
  // Play around with it if you want
  /*rs.addDataPoint(lightSensorValue);
  double avg = rs.getAverage();
  rs.addDataPoint(avg);*/

  // print it to Serial. You can use the Serial Plotter from the Arduino IDE (under "Tools") to see your signal
  Serial.println(lightSensorValue); 
  // Serial.println(avg);

  // Turn on the LED if it is too dark
  if (lightSensorValue < 450){
    digitalWrite(LED_BUILTIN, HIGH);
  }
  else{
    digitalWrite(LED_BUILTIN, LOW);
  }

}
