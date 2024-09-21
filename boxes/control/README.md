# Control

Negative-feedback and servo loops.

----

<details><summary><b>Materials</b></summary><p>

Contents|Level|Description| # |Data|Link|
:-------|:---:|:----------|:-:|:--:|:--:|
H-bridge|10|H-bridge motor driver (SN754410NE)|2|[-D-](_data/datasheets/sn754410.pdf)|[-L-](https://uk.farnell.com/texas-instruments/sn754410ne/ic-peripheral-driver-half-h-1a/dp/3118977)

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

### Grey

1. Wire up your H-bridge to control motor direction.
2. Write a program to control motor speed.

### White

1. Write a program to implement speed and/or position servoing (PID-controller) using the encoder feedback.


----
## NB3

This box will contribute the following (red) components to your NB3

<p align="center">
<img src="_data/images/NB3_control.png" alt="NB3 stage" width="400" height="400">
<p>

----
