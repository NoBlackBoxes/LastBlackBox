# Sensors : Transducers
A sensor converts (transduces) a physical quantity (light, heat, pressure, etc.) into an electrical signal (voltage, current, or resistance).

## [Video](https://vimeo.com/1031477896)

## Concepts
- A transducer converts a physical quantity we care about (light, heat, pressure, etc.) into an electrical signal that we can measure. We will use them as our robot's input sensors.
- Sensors use a range of different mechanisms.
- A piezo can turn pressure into a voltage.
- A photodiode can convert light into a current.
- A number of very common sensors convert (transduce) physical changes into changes in resistance. Such as thermistor, which changes its resistance at different temperatures.
- However, as we will soon learn...computers (and thus our robot) cannot directly measure resistance. They want to measure voltage.
- How do we convert a change in resistance into a change in voltage?
- You already know how to do this.
- ...a voltage divider.
- We simply replace one of our resistors with this sensor material, such as a thermistor...or this device, an LDR, which changes its internal resistance in response to light!

## Connections

## Lesson

- **TASK**: Build a light sensor
    - *Hint*: Build a voltage divider with one of the fixed resistors replaced with an LDR (light-dependent resistor). Does the "output" voltage vary with light level? 
    - *Help*: A guide to completing this task can be found here: [NB3-Building a Light Sensor](https://vimeo.com/??????)
    - *Challenge*: What should the value of the fixed resistor be to maximize the sensitive range of the output voltage for varying light levels?
> Your multimeter should measure a change in voltage as you cover your LDR or shine light on it. The voltage will either increase with more light or decrease, depending on whether your LDR is the first or second resistor in the voltage divider circuit.
