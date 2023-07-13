# The Last Black Box Bootcamp: Day 4 - Digital Signal Processing

## Morning

----

### Light sensor

- Watch this video: [Analog-Digital-Converter](https://www.youtube.com/watch?v=EnfjYwe2A0w&list=PLAROrg3NQn7cyu01HpOv5BWo217XWBZu0&index=29)
- Here is some background for [matplotlib](https://matplotlib.org/stable/tutorials/introductory/quick_start.html)
- *Task 1*: Use the light sensor and the hindbrain (Arduino) to read the light intensity.
- *Task 2*: Use serial write to send the light intensity to the forebrain (Raspberry Pi).
- *Task 3*: Use the Serial Plotter in the Arduino IDE to plot the light intensity.
- *Task 4*: Read the light intensity in Python using the pySerial library.
- *Task 5*: Forward the signal to the host computer using a UDP socket.
- *Task 6*: Plot the light intensity in Python in real-time using the matplotlib library.

### Digital Signal Processing

- *Task 1*: Record audio (e.g. 5 seconds of speech) using the forebrain (Raspberry Pi) and the MEMS microphones.
- *Task 2*: Copy the audio file to your host computer.
- *Task 3*: Plot the audio data across time using matplotlib.
- *Task 4*: Record an audio file where you whistle with a constant tone.

*We need the recordings for the afternoon.*

More information on the tasks are found in a [separate document](Morning.md)

----

## Afternoon

----

### Audio processing

- Live Lecture: 1D signal processing

- *Task 1*: Determine the frequency of the whistling in the audio file using Fourier Transform.
- *Task 2*: Use your forebrain to record audio data and send it to your host computer.
- *Task 3*: Plot the audio data in time and in frequency domain in real-time on your host computer.
- *Task 4* (On your Raspberry Pi): Detect if the determined frequency during whistling is present in the audio data 
- and above a certain threshold.
- *Task 5*: If the frequency is present, let the robot react by going forward or backward.

Idea to task 5: you can create multiple events using `if` statements, such that you can make your robot dance!


### Pro tasks

Depending on your progress, you can tackle the following tasks:

- *Pro task 1*: 

More information on the tasks are found in a [separate document](Afternoon.md)