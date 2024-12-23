# Data

What is a data...for computers? for brains? How can machines decide what to do based on their inputs? A key concept in decision-making is threshold functions. If an input value goes above a threshold, a decision is triggered. How can we make such a trigger out of electronic circuit elements?

Analog-to-Digital...

----

<details><summary><b>Materials</b></summary><p>

Contents|Level|Description| # |Data|Link|
:-------|:---:|:----------|:-:|:--:|:--:|
Comparator|10|LM339 (DIP-14)|2|[-D-](_resources/datasheets/lm2901.pdf)|[-L-](https://uk.farnell.com/texas-instruments/lm2901n/ic-comparator-quad-2901-dip14/dp/3118410)
LED (Red)|10|3 mm/2 mA red LED|2|[-D-](_resources/datasheets/led_HLMP.pdf)|[-L-](https://uk.farnell.com/broadcom-limited/hlmp-1700/led-3mm-red-2-1mcd-626nm/dp/1003207)
LED (Green)|10|3 mm/2 mA green LED|2|[-D-](_resources/datasheets/led_HLMP.pdf)|[-L-](https://uk.farnell.com/broadcom-limited/hlmp-1790/led-3mm-green-2-3mcd-569nm/dp/1003209)
Resistor (330)|10|330 &Omega;/0.25 W|4|[-D-](_resources/datasheets/resistor.pdf)|[-L-](https://uk.farnell.com/multicomp/mf25-2k4/res-2k4-1-250mw-axial-metal-film/dp/9341595)

</p></details>

----

## Topics

- Why digital? - noise, simplicity, speed
  - Can still represent any value (count in binary)
- How digital? Comparator (op-amp with posiitve feedback)
  - Caveats: threshold noise...need hysterisis (Schmitt-Trigger)
- What about analog signals? simple ADC (4-level)...need logic to turn this into 2-bit binary
- Comparaotr is OPEN DRAIN...exaplin or find another comparator

----

## Goals

1. Build a comparator using the LM339N
2. Build a multi-level comparison (A to D)
3. Build a Schmitt-Trigger

----
