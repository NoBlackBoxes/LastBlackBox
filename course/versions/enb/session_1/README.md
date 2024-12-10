# Bootcamp : Session 1 - Electronics

Here we will learn where electricity comes from, how we can measure it, and some cool things we can do with it.

## Introduction

- *Watch this video*: [The Last Black Box : Bootcamp : Introduction](https://vimeo.com/843482137)

## Atoms

> "Let's start at the very beginning, a very good place to start". - *R&H*

- *Watch this video*: [Structure and the Periodic Table](https://vimeo.com/1000458082)
  - When you need it (and you will), then you can find the periodic table [here](../../../boxes/atoms/card/periodic_table.png)

## Electrons

- *Watch this video*: [Voltage](https://vimeo.com/1000730032)
  - ***Task***: Measure the voltage of a AA battery using your Multimeter (select voltage ("V") and touch your probes to either end of the battery)
  - ***Task***: Measure the voltage of 4xAA batteries in series (end to end). *Hint*: You can use your battery holder.
- *Watch this video*: [Conductors](https://vimeo.com/1000740989)
- *Watch this video*: [Current](https://vimeo.com/1000743561)
- *Watch this video*: [Resistors](https://vimeo.com/1000755493)
  - ***Task***: Measure the resistance of some of your resistors

With a voltage source (battery) and resistors, then we can start building "circuits" - complete paths of conduction that allow current to flow from a location with fewer electrons (+) to a location with more electrons (-).
> ***Note***: This is *weird*. Electrons are the things moving. Shouldn't we say that current "flows" from the (-) area to the (+) area? Unfortunately, current was described before anyone knew about electrons and we are stuck with the following awkward convention: **Current is defined to flow from (+) to (-)**...even though we now know that electrons are moving the opposite way.

To help you build electronic circuits, we will assemble a "prototyping platform", which also happens to be the body of your robot (NB3).

- *Watch this video*: [NB3 Body](https://vimeo.com/1005036900)
  - ***Task***: Assemble the robot body (prototyping base board)
    - If you are curious how the *NB3 Body* printed circuit board (PCB) was designed, then you can find the KiCAD files here: [NB3 Body PCB](../../../boxes/electrons/NB3_body)

- *Watch this video*: [Ohm's Law](https://vimeo.com/1000768334)
  - ***Task***: Build the simple circuit below and measure the current flowing when you connect the battery pack. You know the voltage from the batteries (V) and the resistance of the resistor (R). Does Ohm's Law hold?
    - ***Note***: Measuring current with a multimeter is ***tricky***. You can only get an accurate measurement if ***ALL*** of the current in the circuit is forced to flow through your multimeter. This means that when measuring current, your multimeter must be in *series* with the rest of the circuit. (As opposed to measuring voltage, when your multimeter is placed "parallel" to the circuit.)

- *Watch this video*: [Voltage Dividers](https://vimeo.com/1000782478)
  - ***Task***: Build a voltage divider using two resistors of the same value? Measure the intermediate voltage (between the resistors)
  - ***Task***: Build a voltage divider using a variable resistor (potentiometer). Measure the intermediate voltage
    - A guide to completing these tasks can be found here: [Building Voltage Dividers](https://vimeo.com/1000789632)

## Sensors

- *Watch this video*: [Light Sensors](https://vimeo.com/1000794164)
  - ***Task***: Build a light sensor
    - Build a voltage divider with one of the fixed resistors replaced with an LDR (light-dependent resistor). Does the "output" voltage vary with light level? What should the value of the fixed resistor be to maximize the sensitive range of the output voltage for varying light levels?
    - A guide to completing this task (and all the morning circuit building tasks) can be found here: [Building Circuits](https://vimeo.com/1005054579)

## Magnets

- *Watch this video*: [Electromagnetism](https://vimeo.com/1000810115)

## Motors

We will use electromagnets to create the rotational force to move our NB3 robot. 

- *Watch this video*: [DC Motors](https://vimeo.com/1000824116)
- *Watch this video*: [NB3 Muscles](https://vimeo.com/1005039796)
  - ***Task***: Play with your motor. Spin it forwards *and* backwards...

## Transistors

- *Watch this video*: [Semiconductors](https://vimeo.com/1000842810)
- *Watch this video*: [Diodes](https://vimeo.com/1000861996)
  - ***Task***: Illuminate a light-emitting diode (LED). *Remember the current limiting resistor*
- *Watch this video*: [Transistors - MOSFETs](https://vimeo.com/1000873279)
  - ***Task***: Measure the threshold voltage that opens your MOSFET gate. Compare it to the "expected" range listed in the datasheet
    - The datasheet for your MOSFET can be found here: [IRF510](../../../boxes/transistors/_resources/datasheets/IRF510.pdf)

---

# Project
- Build a light-sensitive motor (a motor that spins when the light is ON and stops when the light is OFF...or the other way around).
  - Use the following circuit as a guide:
<p align="center">
<img src="../../../boxes/transistors/_resources/images/MOSFET_motor_driver.png" alt="MOSEFT driver" width="400" height="300">
</p>

> Where do you connect your light sensor?
