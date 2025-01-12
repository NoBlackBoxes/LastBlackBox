# Build a Brain : Digital Decisions
The building blocks of our digital world.

## Transistors
Semiconductors are materials that can both conduct and resist the flow of electrons. You can arrange them such that the conduction of (lots of) electricity can be controlled by a (tiny) external signal. These devices are call **transistors**.

<details><summary><i>Materials</i></summary><p>

Contents|Depth|Description| # |Data|Link|
:-------|:---:|:----------|:-:|:--:|:--:|
MOSFET (10V)|01|Power MOSFET/N-channel (IRF510)|1|[-D-](/boxes/transistors/_resources/datasheets/IRF510.pdf)|[-L-](https://uk.farnell.com/vishay/irf510pbf/mosfet-n-100v-5-6a-to-220ab/dp/1653658)
MOSFET (5V)|01|Power MOSFET/N-channel (IRL510)|1|[-D-](/boxes/transistors/_resources/datasheets/IRL510.pdf)|[-L-](https://uk.farnell.com/vishay/irl510pbf/mosfet-n-logic-to-220/dp/9102779)
Diode|01|IN4001|2|[-D-](/boxes/transistors/_resources/datasheets/IN4001.pdf)|[-L-](https://uk.farnell.com/on-semiconductor/1n4001g/diode-standard-1a-do-41/dp/1458986)
LED (Red)|01|5 mm/2 mA red LED|2|[-D-](/boxes/transistors/_resources/datasheets/led_HLMP.pdf)|[-L-](https://uk.farnell.com/broadcom-limited/hlmp-4700/led-5mm-red-2-3mcd-626nm/dp/1003232)
LED (Green)|01|3 mm/2 mA green LED|2|[-D-](/boxes/transistors/_resources/datasheets/led_HLMP.pdf)|[-L-](https://uk.farnell.com/broadcom-limited/hlmp-1790/led-3mm-green-2-3mcd-569nm/dp/1003209)
Resistor (470)|01|470 &Omega;/0.25 W|2|[-D-](/boxes/transistors/../electrons/_resources/datasheets/resistor.pdf)|[-L-](https://uk.farnell.com/multicomp/mf25-470r/res-470r-1-250mw-axial-metal-film/dp/9341943)

</p></details><hr>

#### Watch this video: [Semiconductors](https://vimeo.com/1032460818)
> We can modify a pure crystal of certain elements (e.g. silicon) to change how well they conduct electricity.


#### Watch this video: [Diodes](https://vimeo.com/1032443724)
> The chemical and electrical equilibrium between charge carriers creates a potential across the PN junction. This junction potential only permits current flow in one direction, which gives **diodes** there delightfully non-linear behavior.

**TASK**: Illuminate a light-emitting diode (LED). *Remember the current limiting resistor!*
<details><summary><strong>Target</strong></summary>
    The LED should only illuminate when installed in one orientation. If you flip it around, then the "diode" of the LED will prevent current flowing through the circuit.
</details><hr>


#### Watch this video: [NB3 : Testing Diodes](https://vimeo.com/1032458879)
> Diodes allow current to flow in only one direction...if you overcome the internal junction potential. Let's measure this...and also use an LED to emit some light.


#### Watch this video: [MOSFETs](https://vimeo.com/1032452466)
> MOSFETs are the thing that humans have built more of than anything else. They must be useful! Let's discuss what they are and how they work.

**TASK**: Measure the threshold voltage that opens your MOSFET gate. Compare it to the "expected" range listed in the
- The datasheet for your MOSFETs can be found here [IRF510](/boxes/transistors/_resources/datasheets/IRF510.pdf) and here [IRL510](/boxes/transistors/_resources/datasheets/IRL510.pdf)
<details><summary><strong>Target</strong></summary>
    The threshold for when current starts to flow through your MOSFET ("Gate-Source Threshold Voltage") should be between 2 to 4 Volts for the IRF510 and 1 to 3 vols for the IRL510. However, the amount of current it allows will rise rapidly up to (and beyond) 10 Volts for the IRF510 and 5 Volts for the IRL510. Check the datasheets (Figure 3).
</details><hr>


# Project
### NB3 : Building a Light-Sensitive Motor
> Let's make something move in response to light!

<details><summary><weak>Guide</weak></summary>
:-:-: A video guide to completing this project can be viewed <a href="https://vimeo.com/1032454998" target="_blank" rel="noopener noreferrer">here</a>.
</details><hr>

**TASK**: Use a MOSFET transistor to control how much current is flowing through your motor and gate the MOSFET with the output voltage from your light sensor...creating a motor that spins when the light is **ON** and stops when the light is **OFF**, or the other way around.
- *Hint*: Use the following circuit as a guide: [MOSFET driver:400](/boxes/transistors/_resources/images/MOSFET_motor_driver.png)
<details><summary><strong>Target</strong></summary>
    Your motor should change how fast it spins when you change how much light hits the LDR.
</details><hr>


