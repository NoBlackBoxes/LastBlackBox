# Control

Negative-feedback and servo loops.

----

<details><summary><b>Materials</b></summary><p>

Contents|Level|Description| # |Data|Link|
:-------|:---:|:----------|:-:|:--:|:--:|
Servo Motor|01|FT90R Digital Micro Continuous Rotation Servo|2|-|[-L-](https://www.pololu.com/product/2817)
Servo Wheel|01|Wheels (70x8mm) for servos|2|-|[-L-](https://www.pololu.com/product/4925)
H-bridge|10|H-bridge motor driver (SN754410NE)|2|[-D-](_resources/datasheets/sn754410.pdf)|[-L-](https://uk.farnell.com/texas-instruments/sn754410ne/ic-peripheral-driver-half-h-1a/dp/3118977)
DC Gearbox Motor|10|TT Gearbox DC Motor - 200RPM - 3 to 6VDC and wheel|2|-|[-L-](https://www.adafruit.com/product/3777#technical-details)

</p></details>

----

## Topics

- (introduce H-bridge) - N and P MOSFETS in an H to control direction of current flowing through motor (and braking)
- (introduce encoders)
- How to control?
- Engine - Watt's Govenor
- P..I..D
- brain?

----

## Goals

1. Wire up your H-bridge to control motor direction.
2. Write a program to control motor speed.
3. Write a program to implement speed and/or position servoing (PID-controller) using the encoder feedback.

----
