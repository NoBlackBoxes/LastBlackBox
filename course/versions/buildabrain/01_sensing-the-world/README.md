# Build a Brain : Sensing the World
We will start by learning how electricity is used to create "sensors", which will be the inputs for your robot brain. Sensors will measure anything you want your robot to know about the world (light intensity, temperature, pressure, sound, etc.).

## Atoms
*"Let's start at the very beginning, a very good place to start"*. - R&H

#### Watch this video: [Atomic Structure](https://vimeo.com/1000458082)
> A brief introduction to the physics of atoms, their parts (protons, neutrons, and electrons), and their classical vs. quantum structure.


#### Watch this video: [The Periodic Table](https://vimeo.com/1028399080)
> Organizing the elements into a table reveals a regular pattern, which is linked to the fundamental chemical properties of each material.

- When you need it *(and you will)*, then you can find a copy of the periodic table [here](/boxes/atoms/_resources/images/periodic_table.png).

#### Watch this video: [Heat](https://vimeo.com/1029691491)
> Atoms in every material are always moving. This motion of atoms (their average kinetic energy) is heat.


## Electrons
Electrons are the sub-atomic particles that underlie *electricity*. Controlling the movement of electrons (and the effects of that movement) will allow us to build many different kinds of electronic devices, from simple circuits to robots and computers.

#### Watch this video: [Voltage](https://vimeo.com/1000730032)
> When there is more negative or positive charge in one location vs. another there is a *potential difference* between these locations. This *potential difference* is called a **voltage** and it creates a pressure that pushes electrons from the location with more negative charge to the location with less.


#### Watch this video: [Conductors](https://vimeo.com/1029337222)
> Some materials have electrons in their outer orbitals that are happy to jump between neighboring atomic nuclei (of the same element). These electrons are "free" to move around the material. If we place such a material between two locations with a *potential difference* (voltage), then electrons will flow from the **(-)** location to the **(+)** location; the material will **conduct** electricity.


#### Watch this video: [Batteries](https://vimeo.com/1029278169)
> Generating a stable voltage requires a renewable source of electrons to maintain a *potential difference*. We can accomplish this with a (redox) chemical reaction inside the wonderfully useful device that we call a **battery**.

**TASK**: Measure the voltage of a AA battery using your multimeter.
- *Hint*: Select the voltage ("V") setting and touch your probes to either end of the battery. Depending on your multimeter, you may also need to select an appropriate "range". For a single AA battery, you should expect to measure between 1 and 2 Volts.
- *Help*: If you are new to using a multimeter, then I recommend that you watch this video: [NB3-Multimeter Basics](https://vimeo.com/1027764019)
- *Help*: If you are new to measuring voltage with a multimeter, then I recommend that you watch this video: [NB3-Measuring Voltage](https://vimeo.com/1027762531)
<details><summary><strong>Target</strong></summary>
:-:-: A single AA battery, fully charged, should have a voltage of ~1.6 Volts. If it is less than 1.5 Volts, then the battery is nearly *dead*.
</details><hr>

**TASK**: Measure the voltage of 4xAA batteries in series (end to end).
- *Hint*: You can use your battery holder.
<details><summary><strong>Target</strong></summary>
:-:-: Batteries connected in series will sum their voltages. You should measure four times the voltage of a single AA battery, ~6.4 Volts, from the batteries in your 4xAA holder.
</details><hr>


#### Watch this video: [Current](https://vimeo.com/1029334167)
> The rate at which electrons flow, measured as *#charges / second*, is called **current**. We use the unit *Amps* (A) and the circuit symbol **I**.


#### Watch this video: [Resistors](https://vimeo.com/1029696806)
> Many materials hold onto their outer electrons and resist their movement. We can create mixtures of these "resisting" materials and better "conducting" materials, often in the form of ceramics, to create **resistors** with a range of different *resistance* values, which we measure in Ohms (&Omega;).

**TASK**: Measure the resistance of your resistors.
- *Help*: If you are new to measuring resistance with a multimeter, then I recommend that you watch this video: [NB3-Measuring Resistance](https://vimeo.com/1027761453)
<details><summary><strong>Target</strong></summary>
:-:-: Your kit contains 470 &Omega;, 1 k&Omega;, and 10 k&Omega; resistors. You should measure these values.
</details><hr>


#### Watch this video: [NB3 : Body](https://vimeo.com/1030776673)
> We will now start measuring and manipulating electricity, but first we will assemble a "prototyping platform" that also happens to be the **body** of your robot (NB3).

**TASK**: Assemble the robot body (prototyping base board).
- *Challenge*: If you are curious how the *NB3 Body* printed circuit board (PCB) was designed, then you can find the KiCAD files here: [NB3 Body PCB](/boxes/electrons/NB3_body).
<details><summary><strong>Target</strong></summary>
:-:-: Your NB3 should now look like [this](/boxes/electrons/NB3_body/NB3_body_front.png). Your breadboards will be different colors...and you should have some rubber feet on the back.
</details><hr>


#### Watch this video: [NB3 : Building Circuits](https://vimeo.com/1030783826)
> With a voltage source (battery) and resistors, then we can start building "circuits" - complete paths of conduction that allow current to flow from a location with *less* electrons **(+)** to a location with *more* electrons **(-)**.

- *Note*: This is *weird*. Electrons are the things moving. Shouldn't we say that current "flows" from the **(-)** area to the **(+)** area? Unfortunately, current was described before anyone knew about electrons and we are stuck with the following awkward convention: **Current is defined to flow from (+) to (-)**...even though we now know that electrons are moving the opposite direction.
**TASK**: Build the simple circuit below and measure the current flowing when you connect the battery pack.
- *Warning*: Measuring current with a multimeter is ***tricky***. You can only get an accurate measurement if ***ALL*** of the current in the circuit is forced to flow through your multimeter. This means that when measuring current, your multimeter must be in *series* with the rest of the circuit. (As opposed to measuring voltage, when your multimeter is placed "parallel" to the circuit.)
- *Help*: If you are new to measuring current with a multimeter, then I recommend you watch this video: [NB3-Measuring Current](https://vimeo.com/1027757287).
<details><summary><strong>Target</strong></summary>
:-:-: Not too much current...and do not break your multimeter.
</details><hr>


#### Watch this video: [Ohm's Law](https://vimeo.com/1029695302)
> Ohm's Law describes the relationship between Voltage, Current, and Resistance. It is not complicated, but it is very useful.

**TASK**: Does Ohm's Law hold? You know the voltage of the batteries (V) and the resistance of the resistor (R). Measure the current flowing (I) for different resistors and confirm that V = I*R.
<details><summary><strong>Target</strong></summary>
:-:-: For the resistors in your kit, then Ohm's Law should determine the current that you measure.
</details><hr>


#### Watch this video: [Voltage Dividers](https://vimeo.com/1030787469)
> Controlling the level of voltage at different places in a circuit is critical to designing electronic devices.

**TASK**: Build a voltage divider using two resistors of the same value. Measure the intermediate voltage (between the resistors).
<details><summary><strong>Target</strong></summary>
:-:-: With equal size resistors, the intermediate voltage you measure should be half of the supply voltage.
</details><hr>

**TASK**: Build a voltage divider using a variable resistor (potentiometer). Measure the intermediate voltage. What happens when you change the position of the internal contact of the variable resistor (by turning the screw)?
- *Help*: A video guide to completing these tasks can be found here: [NB3-Building Voltage Dividers](https://vimeo.com/1030790826)
<details><summary><strong>Target</strong></summary>
:-:-: The intermediate voltage should vary continuously as you adjust the potentiometer.
</details><hr>


## Sensors
Computers and brains work with electrical signals. In order for either to understand signals in the environment (light, sound, pressure, heat, etc.), then these physical signals must be converted into electrical signals. This conversion is called *transduction* and the thing that does it is a *transducer*. However, given their role in sensing the environment, it is common to call these transduction devices ***sensors***.

#### Watch this video: [Transducers](https://vimeo.com/1031477896)
> A sensor converts (transduces) a physical quantity (light, heat, pressure, etc.) into an electrical signal (voltage, current, or resistance).


# Project
### NB3 : Building a Light Sensor
> Your NB3 will use LDRs to convert light into voltage. Here you will build and test this light sensing circuit.

<details><summary><weak>Guide</weak></summary>
:-:-: A video guide to completing this project can be viewed <a href="https://vimeo.com/1031479533" target="_blank" rel="noopener noreferrer">here</a>.
</details><hr>

**TASK**: Build a light sensor
- *Hint*: Build a voltage divider with one of the fixed resistors replaced with an LDR (light-dependent resistor). Does the "output" voltage vary with light level?
- *Challenge*: What should the value of the fixed resistor be to maximize the sensitive range of the output voltage for varying light levels?
<details><summary><strong>Target</strong></summary>
:-:-: Your multimeter should measure a change in voltage as you cover your LDR or shine light on it. The voltage will either increase with more light or decrease, depending on whether your LDR is the first or second resistor in the voltage divider circuit.
</details><hr>

