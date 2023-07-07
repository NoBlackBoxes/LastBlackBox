# The Last Black Box Bootcamp: Day 1 - Electronics

## Morning

----

### Introduction

- Watch this video: [The Last Black Box: Bootcamp - Introduction](https://vimeo.com/625626556)

### NB3 Build (body)

- Watch this video: [NB3 Body](https://vimeo.com/625664801)
- The LastBlackBox GitHub Repo is here: [Last Black Box](https://github.com/NoBlackBoxes/LastBlackBox)
  - The NB3_body PCB design (in KiCAD) is here: [NB3 Body PCB](https://github.com/NoBlackBoxes/LastBlackBox/tree/master/boxes/electrons/NB3_body)

- ***Task 1***: Assemble the robot body (prototyping base board)

### Electricity

- Watch this video: [Atoms, Electons, V=IR, and Sesnors](https://vimeo.com/625820421)

- ***Task 2***: Understand your Multimeter
  - Measure the voltage produced by a single AA battery
  - Measure the volatge of 4xAA batteries in series (in your battery holder).
  - Measure the resistance of some of your resistors (*Bonus:* Is their value consistent with the color coded bands on the resistor?)
- ***Task 3***: Build a voltage divider using discrete resistors
  -  Choose two resistors and construct a divider circuit. Does the voltage at the "middle" make sense given the two resistors your used?
- ***Task 4***: Build a variable voltage divider using a potentionmeter
  - Use the reistance function of your Multimeter to measure the resistance of the potentionmeter (pot) across different pairs of its three pins (legs). Which pins show a varying resistance and which pins are stable as you turn the dial? The pin that varies is the one connected to the rotary screw. (*Clue*: It's the middle one.)
  - Turn your pot to an intermediate position...away from either extreme...and connect your batteries (+ and -) to the other pins. (***Caution***: If you connect your batteries across a pot turned to a very low resistance, then a **lot** of current will try to flow. This can cause your pot to heat up...and maybe even break....start somewhere in the middle.)
  - Measure the voltage at the variable pin (relative to ground). Watch (*hopefully*) the output voltage vary as your rotate the position of your "variable reisistor" (pot). You just built a rotation sensor. :)
- ***Task 5***: Build a light sensor
  - Build a voltage divider with one of the fixed resistors replaced with an LDR (light-dependent resistor). Does the "output" voltage vary with light? What should the value of the fixed resistor be to maximize the sensitivity of the output voltage varying light levels?

### Magnetism

- Watch this video: [Magnets, Electromagnetism, and DC Motors](https://vimeo.com/626603421)
- ***Task 6***: Spin your motor (forwards and backwards)...just flip the + and - connections.

----

## Afternoon

----

### Transistors

- Live Lecture: "Semiconductors, diodes, and transistors"
- ***Task 1***: Build a MOSFET motor driver circuit


<p align="center">
<img src="resources/images/MOSFET_motor_driver.png" alt="MOSEFT driver" width="400" height="300">
</p>

- ***Project***: Build light sensitive motor!

## Evening

### NB3 Build (servos)

- Watch this video: [NB3 Servos](https://vimeo.com/625827358)
- ***Task 1***: Mount the robot servo motors and wheels


----
