# The Last Black Box : *Bootcamp* : Day 1 - Electronics

----------
## Morning

### Introduction

- *Watch this video*: [The Last Black Box : Bootcamp : Introduction](https://vimeo.com/843482137)

#### Atoms

> "Let's start at the very beginning, a very good place to start". - *R&H*

- *Watch this video*: [Atoms: Structure and the Periodic Table](https://vimeo.com/1000458082)
  - When you need it (and you will), then you can find the periodic table [here](../../../boxes/atoms/card/periodic_table.png)

#### Electrons

- *Watch this video*: [Voltage](https://vimeo.com/1000730032)
- *Watch this video*: [Batteries](https://vimeo.com/XXXXXXXXX)
  - ***Task***: Measure the voltage of a AA battery
  - ***Task***: Measure the voltage of 4 AA batteries in series (end to end)
  - If you a new to "multimeters", then watch this video for an introduction to measuring voltage: [Multimeters - Voltage](https://vimeo.com/XXXXXXXXX)

- *Watch this video*: [Conductors](https://vimeo.com/1000740989)
- *Watch this video*: [Current](https://vimeo.com/1000743561)
- *Watch this video*: [Resistors](https://vimeo.com/XXXXXXXXX)
  - ***Task***: Measure the resistance of some of your resistors
    - If you are new to "multimeters", then watch this video for an introduction to measuring resistance: [Multimeters - Resistance](https://vimeo.com/XXXXXXXXX)

With a voltage source (battery) and resistors, then we can start building "circuits" - complete paths of conduction that allow current to from a location with fewer electrons (+) to a location with more electrons (-).
> ***Note***: This is *weird*. Electrons are the things moving. Shouldn't we say that current "flows" from the (-) area to the (+) area? Unfortunately, current (I) was described before anyone knew about electrons and we are stuck with the following awkward convention: **Current is defined to flow from (+) to (-)**...even though we now know that electrons are moving the opposite way.

To help you build electornic circuits, we will assemble a "prototyping platform", which also happens to be the body of your robot (NB3).

- *Watch this video*: [NB3 Body](https://vimeo.com/843622939)
  - ***Task***: Assemble the robot body (prototyping base board)
    - If you are curious how the *NB3 Body* printed circuit board (PCB) was designed, then you can find the KiCAD files here: [NB3 Body PCB](../../../boxes/electrons/NB3_body)

- *Watch this video*: [Ohm's Law](https://vimeo.com/XXXXXXXXX)
  - ***Task***: Build the simple circuit below and measure the current flowing when you connect the battery pack. You know the voltage from the batteries (V) and the resistance of the resistor (R). Does Ohm's Law hold?
    - ***Note***: Measuring current with a multimeter is tricky. If you are not ***super*** confident that you know what you are doing, then I encourage you to watch this video for an introduction to measuring current: [Multimeters - Current](https://vimeo.com/XXXXXXXXX)

- *Watch this video*: [Voltage Dividers](https://vimeo.com/XXXXXXXXX)
  - ***Task***: Build a voltage divider using two resistors of the same value? Measure the intermediate voltage
  - ***Task***: Build a voltage divider using a variable resistor (potentiometer). Measure the intermediate voltage
    - A guide to completing these tasks can be found here: [Building Voltage Dividers](https://vimeo.com/XXXXXXXXX)

#### Sensors

- *Watch this video*: [Light Sensors](https://vimeo.com/XXXXXXXXX)
  - ***Task***: Build a light sensor
    - Build a voltage divider with one of the fixed resistors replaced with an LDR (light-dependent resistor). Does the "output" voltage vary with light level? What should the value of the fixed resistor be to maximize the sensitive range of the output voltage for varying light levels?

#### Magnets

- *Watch this video*: [Magnetism](https://vimeo.com/XXXXXXXXX)
  - ***Bonus (not required) Task***: Build a speaker
    - If you have access to some thin wire (ideally "magnet" wire), a paper cup, tape, and an audio source (headphone jack on your phone/laptop), then it is relatively easy to build a speaker. If you are curious how this is done, then you can watch this video: [Build a Speaker](https://vimeo.com/XXXXXXXXX)

#### Motors

We will use electromagnets to create the rotational force to move our NB3 robot. 

- *Watch this video*: [NB3 Motors](https://vimeo.com/843634014)
  - ***Task***: Spin your motor (forwards and backwards...just flip the + and - connections)

------------
## Afternoon

#### Transistors

- *Watch this video*: [Semiconductors](https://vimeo.com/XXXXXXXXX)
  - ***Task***: Identify alternative doping combinations

- *Watch this video*: [Diodes](https://vimeo.com/XXXXXXXXX)
  - ***Task***: Confirm that diodes only pass current in one direction
  - ***Task***: Measure the voltage drop across a diode (the forward threshold)
  - ***Task***: Illuminate a light-emitting diode (LED). *Remember the current limiting resistor*

- *Watch this video*: [Transistors - MOSFETs](https://vimeo.com/XXXXXXXXX)
  - ***Task***: Measure the threshold voltage that opens your MOSFET gate. Compare it to the "expected" range listed in the datasheet
    - The datasheet for your MOSFET can be found here: [IRF510](../../../boxes/transistors/_data/datasheets/IRF510.pdf)

### PROJECT
- Build a light-sensitive motor (a motor that spins when the light is ON and stops when the light is OFF...or the other way around).
  - Use the following circuit as a guide:
<p align="center">
<img src="../../../boxes/transistors/_data/images/MOSFET_motor_driver.png" alt="MOSEFT driver" width="400" height="300">
</p>

> Where do you connect your light sensor?

----
