# transistors

The signals arriving from the environment are often too small to measure reliably with a computer or brain. The siganls they generate are often too small to act with the force necessary to sufficiently change the environment. We need to make them bigger, for both more reliable input and more effective output. Making tiny voltage or current signals larger is called *amplification*, and the things that do this are ***amplifiers***.

----

<details><summary><b>Materials</b></summary><p>

Contents|Description| # |Data|Link|
:-------|:----------|:-:|:--:|:--:|
MOSFET|Power MOSFET/N-channel (IRF510)|2|[-D-](_data/datasheets/IRF510.pdf)|[-L-](https://uk.farnell.com/vishay/irf510pbf/mosfet-n-100v-5-6a-to-220ab/dp/1653658)
Diode|IN401|2|[-D-](_data/datasheets/IN4001.pdf)|[-L-](https://uk.farnell.com/on-semiconductor/1n4001g/diode-standard-1a-do-41/dp/1458986)
LED(blue)|Low power blue light emitting diode|2|[-D-](_data/datasheets/LED_blue.pdf)|[-L-](https://uk.farnell.com/broadcom-limited/hlmp-ka45-e0000/led-3mm-blue-85mcd-470nm/dp/1863182)

Required|Description| # |Box|
:-------|:----------|:-:|:-:|
Multimeter|(Sealy MM18) pocket digital multimeter|1|[white](/boxes/white/README.md)|
Test Lead|Alligator clip to 0.64 mm pin (20 cm)|2|[white](/boxes/white/README.md)|
Batteries (AA)|AA 1.5 V alkaline battery|4|[electrons](/boxes/electrons/README.md)|
Battery holder|4xAA battery holder with ON-OFF switch|1|[electrons](/boxes/electrons/README.md)|
Jumper kit|Kit of multi-length 22 AWG breadboard jumpers|1|[electrons](/boxes/electrons/README.md)|
Jumper wires|Assorted 22 AWG jumper wire leads (male/female)|1|[electrons](/boxes/electrons/README.md)|

</p></details>

----

## Goals

To open this box, you must complete the following tasks:

1. Confirm diodes
2. Measure the threshold voltage of a MOSFET (transistor)
3. Build a a light sensitive motor (with protection diode)

To explore this box, you should attempt the following challenges:

1. The world's worst audio amplifier?

----

## NB3

This box will contribute the following (red) components to your NB3

<p align="center">
<img src="_images/NB3_amplifiers.png" alt="NB3 stage" width="400" height="400">
<p>

----

## MOSEFT motor driver circuit (with protection diode)

<p align="center">
<img src="_images/MOSFET_motor_driver.png" alt="MOSFET Motor Driver" width="800" height="600">
<p>

## Tubes

Amplifying electrical signals had been a long sought goal. The early attempts had a major impact on technology, but they had many drawbacks.

...a little like light bulbs.

### Exercise: ???

- Tubes!

----

## Semiconductors

A clue.

<p align="center">
<img src="_images/silicon.png" alt="Silicon" width="150" height="150">
<p>

### Exercise: Wafer

- Look at it.

----

## Diodes

Why are they useful (IV kink)

<p align="center">
<img src="_images/on_junction.png" alt="PN Junction" width="150" height="150">
<p>

### Exercise: AM radio

- Do some rectification...

----

## Transistors

Solid-state amplifiers!

<p align="center">
<img src="_images/fet.png" alt="Field-Effect Transistor" width="150" height="150">
<p>

### Exercise: Modulate a motor with light

- First steps towards a vehicle!
- Don't forget the protection diode!!

----
