# Robotics : NB3 : Build a Braitenberg Vehicle
Here we create the first Braitenberg Vehicle, a simple sensory-motor feedback loop connecting two light sensors to the motion of two wheels.

## [Video](https://vimeo.com/1034798460)
- Describe Braitenberg Vehicles
- Design overview of system
- Add two light sensors using NB3 body connections
- Test sensors
- Create behavior loop
- Test

## Concepts

## Connections

## Lesson
- A Braitenberg Vehicle can show complex behaviour, appearing to seek out light and avoid shadows, but the underlying control circuit is extremely simple.
- ![Braitenberg Vehicle:600](/boxes/robotics/_resources/images/braitenberg_vehicle.png)
- A small change to the control circuit can completely change how your NB£ "vehicle" responds to light.

- **TASK**: Measure two light sensors and *decide* how to activate the direction of your two wheels in response.
- Some example code to get you started can be found here: [Braitenberg Vehicle (Arduino)](/boxes/robotics/programming/arduino/braitenberg_vehicle/braitenberg_vehicle.ino)
> You should have created a robot that either likes (turns toward) or avoids (turns away from) light.
