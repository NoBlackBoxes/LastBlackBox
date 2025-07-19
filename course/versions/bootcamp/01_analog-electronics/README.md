# Bootcamp : Analog Electronics
Here we will learn where electricity comes from, how we can measure it, and what we can do with it.

## Atoms
*"Let's start at the very beginning, a very good place to start"*. - R&H

<details><summary><i>Materials</i></summary><p>

Name|Depth|Description| # |Data|Link|
:-------|:---:|:----------|:-:|:--:|:--:|
Periodic Table|01|Periodic Table business card|1|[-D-](/boxes/atoms/card)|[-L-](VK)

</p></details><hr>

#### Watch this video: 
<p align="center">
<a href="https://vimeo.com/1000458082" title="Control+Click to watch in new tab"><img src="../../../../boxes/atoms/_resources/lessons/thumbnails/Atomic-Structure.gif" alt="Atomic Structure" width="480"/></a>
</p>

> A brief introduction to the physics of atoms, their parts (protons, neutrons, and electrons), and their classical vs. quantum structure.


#### Watch this video: 
<p align="center">
<a href="https://vimeo.com/1028399080" title="Control+Click to watch in new tab"><img src="../../../../boxes/atoms/_resources/lessons/thumbnails/The-Periodic-Table.gif" alt="The Periodic Table" width="480"/></a>
</p>

> Organizing the elements into a table reveals a regular pattern, which is linked to the fundamental chemical properties of each material.

- When you need it *(and you will)*, then you can find a copy of the periodic table [here](/boxes/atoms/_resources/images/periodic_table.png).

## Electrons
Electrons are the sub-atomic particles that underlie *electricity*. Controlling the movement of electrons (and the effects of that movement) will allow us to build many different kinds of electronic devices, from simple circuits to robots and computers.

<details><summary><i>Materials</i></summary><p>

Name|Depth|Description| # |Data|Link|
:-------|:---:|:----------|:-:|:--:|:--:|
NB3 Body|01|NB3 robot base PCB|1|[-D-](/boxes/electrons/NB3_body)|[-L-](VK)
Resistor (470)|01|470 &Omega;/0.25 W|2|[-D-](/boxes/electrons/_resources/datasheets/resistor.pdf)|[-L-](https://uk.farnell.com/multicomp/mf25-470r/res-470r-1-250mw-axial-metal-film/dp/9341943)
Resistor (1k)|01|1 k&Omega;/0.25 W|2|[-D-](/boxes/electrons/_resources/datasheets/resistor.pdf)|[-L-](https://uk.farnell.com/multicomp/mf25-1k/res-1k-1-250mw-axial-metal-film/dp/9341102)
Resistor (10k)|01|10 k&Omega;/0.25 W|2|[-D-](/boxes/electrons/_resources/datasheets/resistor.pdf)|[-L-](https://uk.farnell.com/multicomp/mf25-10k/res-10k-1-250mw-axial-metal-film/dp/9341110)
Potentiometer|01|2.2 k&Omega; variable resistor|2|[-D-](/boxes/electrons/_resources/datasheets/pot_2k2.pdf)|[-L-](https://uk.farnell.com/bourns/3362p-1-222lf/trimmer-pot-2-2kohm-10-1turn-th/dp/2328599)
Breadboard (400)|01|400-tie solderless breadboard|1|[-D-](/boxes/electrons/_resources/datasheets/breadboard_400.pdf)|[-L-](https://www.amazon.co.uk/gp/product/B0739XRX8F)
Breadboards (170)|01|170-tie solderless breadboard|4|[-D-](/boxes/electrons/_resources/datasheets/breadboard_170.pdf)|[-L-](https://www.amazon.co.uk/ELEGOO-tie-points-Breadboard-Breadboards-Electronic/dp/B01N0YWIR7)
Batteries (AA)|01|AA 1.5 V alkaline battery|4|[-D-](/boxes/electrons/)|[-L-](https://www.amazon.co.uk/Duracell-Optimum-Alkaline-Batteries-MX1500/dp/B093Q5XY66)
Battery holder|01|4xAA battery holder with ON-OFF switch|1|[-D-](/boxes/electrons/)|[-L-](https://www.amazon.co.uk/gp/product/B0814ZH68F)
Jumper Kit|01|Kit of multi-length 22 AWG breadboard jumpers|1|[-D-](/boxes/electrons/_resources/datasheets/jumper_kit.pdf)|[-L-](https://uk.farnell.com/multicomp/mc001810/hard-jumper-wire-22awg-140pc/dp/2770338)
Jumper Wires|01|Assorted 22 AWG jumper wire leads (male/female)|1|[-D-](/boxes/electrons/_resources/datasheets/jumper_wires.pdf)|[-L-](https://www.amazon.co.uk/gp/product/B09KR7Z4PF)
Test Lead|01|Alligator clip to 0.64 mm pin (20 cm)|2|[-D-](/boxes/electrons/)|[-L-](https://www.amazon.co.uk/gp/product/B096JR15JW)
Rubber feet|01|Adhesive rubber standoffs (1421T6CL)|4|[-D-](/boxes/electrons/_resources/datasheets/rubber_feet.pdf)|[-L-](https://uk.farnell.com/hammond/1421t6cl/feet-stick-on-pk24/dp/1876522)

</p></details><hr>

#### Watch this video: 
<p align="center">
<a href="https://vimeo.com/1000730032" title="Control+Click to watch in new tab"><img src="../../../../boxes/electrons/_resources/lessons/thumbnails/Voltage.gif" alt="Voltage" width="480"/></a>
</p>

> When there is more negative or positive charge in one location vs. another there is a *potential difference* between these locations. This *potential difference* is called a **voltage** and it creates a pressure that pushes electrons from the location with more negative charge to the location with less.


#### Watch this video: 
<p align="center">
<a href="https://vimeo.com/1029337222" title="Control+Click to watch in new tab"><img src="../../../../boxes/electrons/_resources/lessons/thumbnails/Conductors.gif" alt="Conductors" width="480"/></a>
</p>

> Some materials have electrons in their outer orbitals that are happy to jump between neighboring atomic nuclei (of the same element). These electrons are "free" to move around the material. If we place such a material between two locations with a *potential difference* (voltage), then electrons will flow from the **(-)** location to the **(+)** location; the material will **conduct** electricity.


#### Watch this video: 
<p align="center">
<a href="https://vimeo.com/1029278169" title="Control+Click to watch in new tab"><img src="../../../../boxes/electrons/_resources/lessons/thumbnails/Batteries.gif" alt="Batteries" width="480"/></a>
</p>

> Generating a stable voltage requires a renewable source of electrons to maintain a *potential difference*. We can accomplish this with a (redox) chemical reaction inside the wonderfully useful device that we call a **battery**.


**TASK**: Measure the voltage of a AA battery using your multimeter.
- - *Hint*: Select the voltage ("V") setting and touch your probes to either end of the battery. Depending on your multimeter, you may also need to select an appropriate "range". For a single AA battery, you should expect to measure between 1 and 2 Volts.
- - *Help*: If you are new to using a multimeter, then I recommend that you watch this video: [NB3-Multimeter Basics](https://vimeo.com/1027764019)
- - *Help*: If you are new to measuring voltage with a multimeter, then I recommend that you watch this video: [NB3-Measuring Voltage](https://vimeo.com/1027762531)
<details><summary><strong>Target</strong></summary>
    A single AA battery, fully charged, should have a voltage of ~1.6 Volts. If it is less than 1.5 Volts, then the battery is nearly *dead*.
</details><hr>


**TASK**: Measure the voltage of 4xAA batteries in series (end to end).
- - *Hint*: You can use your battery holder.
<details><summary><strong>Target</strong></summary>
    Batteries connected in series will sum their voltages. You should measure four times the voltage of a single AA battery, ~6.4 Volts, from the batteries in your 4xAA holder.
</details><hr>


#### Watch this video: 
<p align="center">
<a href="https://vimeo.com/1029334167" title="Control+Click to watch in new tab"><img src="../../../../boxes/electrons/_resources/lessons/thumbnails/Current.gif" alt="Current" width="480"/></a>
</p>

> The rate at which electrons flow, measured as *#charges / second*, is called **current**. We use the unit *Amps* (A) and the circuit symbol **I**.


#### Watch this video: 
<p align="center">
<a href="https://vimeo.com/1029696806" title="Control+Click to watch in new tab"><img src="../../../../boxes/electrons/_resources/lessons/thumbnails/Resistors.gif" alt="Resistors" width="480"/></a>
</p>

> Many materials hold onto their outer electrons and resist their movement. We can create mixtures of these "resisting" materials and better "conducting" materials, often in the form of ceramics, to create **resistors** with a range of different *resistance* values, which we measure in Ohms (&Omega;).


**TASK**: Measure the resistance of your resistors.
- - *Help*: If you are new to measuring resistance with a multimeter, then I recommend that you watch this video: [NB3-Measuring Resistance](https://vimeo.com/1027761453)
<details><summary><strong>Target</strong></summary>
    Your kit contains 470 &Omega;, 1 k&Omega;, and 10 k&Omega; resistors. You should measure these values.
</details><hr>


#### Watch this video: 
<p align="center">
<a href="https://vimeo.com/1030776673" title="Control+Click to watch in new tab"><img src="../../../../boxes/electrons/_resources/lessons/thumbnails/NB3_:-Body.gif" alt="NB3 : Body" width="480"/></a>
</p>

> We will now start measuring and manipulating electricity, but first we will assemble a "prototyping platform" that also happens to be the **body** of your robot (NB3).


**TASK**: Assemble the robot body (prototyping base board).
- - *Challenge*: If you are curious how the *NB3 Body* printed circuit board (PCB) was designed, then you can find the KiCAD files here: [NB3 Body PCB](/boxes/electrons/NB3_body).
<details><summary><strong>Target</strong></summary>
    Your NB3 should now look like [this](/boxes/electrons/NB3_body/NB3_body_front.png). Your breadboards will be different colors...and you should have some rubber feet on the back.
</details><hr>


#### Watch this video: 
<p align="center">
<a href="https://vimeo.com/1030783826" title="Control+Click to watch in new tab"><img src="../../../../boxes/electrons/_resources/lessons/thumbnails/NB3_:-Building-Circuits.gif" alt="NB3 : Building Circuits" width="480"/></a>
</p>

> With a voltage source (battery) and resistors, then we can start building "circuits" - complete paths of conduction that allow current to flow from a location with *less* electrons **(+)** to a location with *more* electrons **(-)**.

- *Note*: This is *weird*. Electrons are the things moving. Shouldn't we say that current "flows" from the **(-)** area to the **(+)** area? Unfortunately, current was described before anyone knew about electrons and we are stuck with the following awkward convention: **Current is defined to flow from (+) to (-)**...even though we now know that electrons are moving the opposite direction.

**TASK**: Build the simple circuit below and measure the current flowing when you connect the battery pack.
- - *Warning*: Measuring current with a multimeter is ***tricky***. You can only get an accurate measurement if ***ALL*** of the current in the circuit is forced to flow through your multimeter. This means that when measuring current, your multimeter must be in *series* with the rest of the circuit. (As opposed to measuring voltage, when your multimeter is placed "parallel" to the circuit.)
- - *Help*: If you are new to measuring current with a multimeter, then I recommend you watch this video: [NB3-Measuring Current](https://vimeo.com/1027757287).
<details><summary><strong>Target</strong></summary>
    Not too much current...and do not break your multimeter.
</details><hr>


#### Watch this video: 
<p align="center">
<a href="https://vimeo.com/1029695302" title="Control+Click to watch in new tab"><img src="../../../../boxes/electrons/_resources/lessons/thumbnails/Ohm's-Law.gif" alt="Ohm's Law" width="480"/></a>
</p>

> Ohm's Law describes the relationship between Voltage, Current, and Resistance. It is not complicated, but it is very useful.


**TASK**: Does Ohm's Law hold? You know the voltage of the batteries (V) and the resistance of the resistor (R). Measure the current flowing (I) for different resistors and confirm that V = I*R.
<details><summary><strong>Target</strong></summary>
    For the resistors in your kit, then Ohm's Law should determine the current that you measure.
</details><hr>


#### Watch this video: 
<p align="center">
<a href="https://vimeo.com/1030787469" title="Control+Click to watch in new tab"><img src="../../../../boxes/electrons/_resources/lessons/thumbnails/Voltage-Dividers.gif" alt="Voltage Dividers" width="480"/></a>
</p>

> Controlling the level of voltage at different places in a circuit is critical to designing electronic devices.


**TASK**: Build a voltage divider using two resistors of the same value. Measure the intermediate voltage (between the resistors).
<details><summary><strong>Target</strong></summary>
    With equal size resistors, the intermediate voltage you measure should be half of the supply voltage.
</details><hr>


**TASK**: Build a voltage divider using a variable resistor (potentiometer). Measure the intermediate voltage. What happens when you change the position of the internal contact of the variable resistor (by turning the screw)?
- - *Help*: A video guide to completing these tasks can be found here: [NB3-Building Voltage Dividers](https://vimeo.com/1030790826)
<details><summary><strong>Target</strong></summary>
    The intermediate voltage should vary continuously as you adjust the potentiometer.
</details><hr>


## Sensors
Computers and brains work with electrical signals. In order for either to understand signals in the environment (light, sound, pressure, heat, etc.), then these physical signals must be converted into electrical signals. This conversion is called *transduction* and the thing that does it is a *transducer*. However, given their role in sensing the environment, it is common to call these transduction devices ***sensors***.

<details><summary><i>Materials</i></summary><p>

Name|Depth|Description| # |Data|Link|
:-------|:---:|:----------|:-:|:--:|:--:|
Photoresistor (LDR)|01|Light-dependent resistor (GL5516 and GL5528)|4|[-D-](/boxes/sensors/)|[-L-](https://uk.farnell.com/advanced-photonix/nsl-19m51/light-dependent-resistor-550nm/dp/3168335)

</p></details><hr>

#### Watch this video: 
<p align="center">
<a href="https://vimeo.com/1031477896" title="Control+Click to watch in new tab"><img src="../../../../boxes/sensors/_resources/lessons/thumbnails/Transducers.gif" alt="Transducers" width="480"/></a>
</p>

> A sensor converts (transduces) a physical quantity (light, heat, pressure, etc.) into an electrical signal (voltage, current, or resistance).


# Project
### NB3 : Building a Light Sensor
> Your NB3 will use LDRs to convert light into voltage. Here you will build and test this light sensing circuit.

<details><summary><weak>Guide</weak></summary>
:-:-: A video guide to completing this project can be viewed <a href="https://vimeo.com/1031479533" target="_blank" rel="noopener noreferrer">here</a>.
</details><hr>


**TASK**: Build a light sensor
- - *Hint*: Build a voltage divider with one of the fixed resistors replaced with an LDR (light-dependent resistor). Does the "output" voltage vary with light level?
- - *Challenge*: What should the value of the fixed resistor be to maximize the sensitive range of the output voltage for varying light levels?
<details><summary><strong>Target</strong></summary>
    Your multimeter should measure a change in voltage as you cover your LDR or shine light on it. The voltage will either increase with more light or decrease, depending on whether your LDR is the first or second resistor in the voltage divider circuit.
</details><hr>


