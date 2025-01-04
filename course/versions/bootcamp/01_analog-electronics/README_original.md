# Bootcamp : Analog Electronics
Here we will learn where electricity comes from, how we can measure it, and what we can do with it.

## Atoms
*"Let's start at the very beginning, a very good place to start"*. - R&H

#### Watch this video: [Atomic Structure](https://vimeo.com/1000458082)
> A brief introduction to the physics of atoms, their parts (protons, neutrons, and electrons), and their classical vs. quantum structure.

#### Watch this video: [The Periodic Table](https://vimeo.com/1000458082)
> The organization of elements into a table reveals a regular pattern, which is linked to the fundamental chemical properties of each material.

- When you need it *(and you will)*, then you can find a copy of the periodic table [here](../../../boxes/atoms/_resources/images/periodic_table.png).
- The electron configuration (assignments to specific orbitals) of each atomic element can be viewed [here](https://en.wikipedia.org/wiki/Electron_configurations_of_the_elements_(data_page)). 

## Electrons
Electrons are the sub-atomic particles that underlie *electricity*. Controlling the movement of electrons (and the effects of their movement) will allow us to build many different kinds of electronic devices, from simple circuits to robots and computers.

#### Watch this video: [Voltage](https://vimeo.com/1000730032)
> When there is more negative charge in one location vs. another, we say there is a *potential difference* between these locations. This *potential difference* is called a **voltage** and it creates a force pushing electrons from the location with more negative charge to the location with less.

#### Watch this video: [Conductors](https://vimeo.com/1000740989)
> Some materials have electrons way out in their outer orbitals that are happy to jump between neighboring atomic nuclei (of the same element). We say that these electrons are "free" to move around the material. If we place such a material between two locations with a *potential difference* (voltage), then electrons will flow from the **(-)** location to the **(+)** location; the material will **conduct** electricity. 

#### Watch this video: [Batteries](https://vimeo.com/??????)
> Generating a stable voltage requires a source of electrons to maintain the *potential difference*, even when electrons are able to move between two areas of different charge. We can accomplish this with a (redox) chemical reaction inside the wonderfully useful device we call a **battery**.

- [ ] **TASK**: Measure the voltage of a AA battery using your multimeter.
- *Hint*: Select the voltage ("V") setting and touch your probes to either end of the battery. Depending on your multimeter, you may also need to select an appropriate "range". For a single AA battery, you should expect to measure between 1 and 2 Volts.
- *Help*: If you are new to measuring voltage with a multimeter, then I recommend you watch this video: [NB3-Measuring Voltage](https://vimeo.com/??????)
<details><summary><strong>Target</strong></summary>
A single AA battery, fully charged, should have a voltage of ~1.6 Volts. If it is less than 1.5 Volts, then the battery is nearly *dead*.
</details><hr>

- [ ] **TASK**: Measure the voltage of 4xAA batteries in series (end to end).
- *Hint*: You can use your battery holder.
<details><summary><strong>Target</strong></summary>
Batteries connected in series will sum their voltages. You should measure four times the voltage of a single AA battery, ~6.4 Volts, from the batteries in your 4xAA holder.
</details><hr>

#### Watch this video: [Current](https://vimeo.com/1000743561)
> The rate at which electrons flow, measured as *#charges / second*, is called **current**. We use the unit *Amps* (A).

#### Watch this video: [Resistors](https://vimeo.com/1000755493)
> Many materials hold onto their outer electrons and resist their movement. We can create mixtures of these "resisting" materials and better "conducting" materials, often in the form of ceramics, to create **resistors** with a range of different *resistance* values, which we measure in Ohms (&Omega;).

- [ ] **TASK**: Measure the resistance of your resistors.
- *Help*: If you are new to measuring resistance with a multimeter, then I recommend you watch this video: [NB3-Measuring Resistance](https://vimeo.com/??????)
<details><summary><strong>Target</strong></summary>
Your kit contains 470 &Omega;, 1 k&Omega;, and 10 k&Omega; resistors. You should measure these values.
</details><hr>

#### Watch this video: [NB3-Body](https://vimeo.com/1005036900)
> To help you start measuring and manipulating electricity, we will first assemble a "prototyping platform", which also happens to be the **body** of your robot (NB3).

- [ ] **TASK**: Assemble the robot body (prototyping base board).
- *Challenge*: If you are curious how the *NB3 Body* printed circuit board (PCB) was designed, then you can find the KiCAD files here: [NB3 Body PCB](../../../boxes/electrons/NB3_body). You can also watch this short introduction to PCB design with KiCAD here: [NB3-Designing PCBs with KiCAD](https://vimeo.com/??????).
<details><summary><strong>Target</strong></summary>
Your NB3 should now look like [this](../../../boxes/electrons/NB3_body/NB3_body_front.png). Your breadboards will be different colors...and you should have some rubber feet on the back.
</details><hr>

#### Watch this video: [NB3-Building Circuits](https://vimeo.com/??????)
> With a voltage source (battery) and resistors, then we can start building "circuits" - complete paths of conduction that allow current to flow from a location with *less* electrons **(+)** to a location with *more* electrons **(-)**.

> ***Note***: This is *weird*. Electrons are the things moving. Shouldn't we say that current "flows" from the **(-)** area to the **(+)** area? Unfortunately, current was described before anyone knew about electrons and we are stuck with the following awkward convention: **Current is defined to flow from (+) to (-)**...even though we now know that electrons are moving the opposite direction.

- [ ] **TASK**: Build the simple circuit below and measure the current flowing when you connect the battery pack.
- ***Warning!***: Measuring current with a multimeter is ***tricky***. You can only get an accurate measurement if ***ALL*** of the current in the circuit is forced to flow through your multimeter. This means that when measuring current, your multimeter must be in *series* with the rest of the circuit. (As opposed to measuring voltage, when your multimeter is placed "parallel" to the circuit.)
- If you are new to measuring current with a multimeter, then I recommend you watch this video: [NB3-Measuring Current](https://vimeo.com/??????).
<details><summary><strong>Target</strong></summary>
Not too much current...and do not break your meter.
</details><hr>

#### Watch this video: [Ohm's Law](https://vimeo.com/1000768334)
> Ohm's Law describes the relationship between Voltage, Current, and Resistance. It is not complicated, but it is very useful.

- [ ] **TASK**: Does Ohm's Law hold? You know the voltage of the batteries (V) and the resistance of the resistor (R). Measure the current flowing (I) for different resistors and confirm that V = I*R.
<details><summary><strong>Target</strong></summary>
For the resistors in your kit, then Ohm's Law should determine the current you measure.
</details><hr>

#### Watch this video: [Voltage Dividers](https://vimeo.com/1000782478)
> Controlling the level of voltage at different places in a circuit is critical to designing electronic devices.

- [ ] **TASK**: Build a voltage divider using two resistors of the same value. Measure the intermediate voltage (between the resistors).
<details><summary><strong>Target</strong></summary>
With equal size resistors, the intermediate voltage you measure should be half of the supply voltage.
</details><hr>

- [ ] **TASK**: Build a voltage divider using a variable resistor (potentiometer). Measure the intermediate voltage. What happens when you change the position of the internal contact of the variable resistor (by turning the screw)?
- *Help*: A video guide to completing these tasks can be found here: [NB3-Building Voltage Dividers](https://vimeo.com/1000789632)
<details><summary><strong>Target</strong></summary>
The intermediate voltage should vary continuously as you adjust the potentiometer.
</details><hr>

## Sensors
Your robot's brain is based on a digital computer that can *only* measure electrical signals. Sensors are needed to convert (transduce) physical signals (light intensity, heat, air pressure, etc.) into electrical signals.

#### Watch this video: [Light Sensors](https://vimeo.com/1000794164)
> A light sensor converts light intensity into an electrical signal (voltage, current, or resistance).

- **TASK**: Build a light sensor
- *Hint*: Build a voltage divider with one of the fixed resistors replaced with an LDR (light-dependent resistor). Does the "output" voltage vary with light level? 
- *Help*: A guide to completing this task can be found here: [NB3-Building a Light Sensor](https://vimeo.com/??????)
- *Challenge*: What should the value of the fixed resistor be to maximize the sensitive range of the output voltage for varying light levels?
<details><summary><strong>Target</strong></summary>
Your multimeter should measure a change in voltage as you cover your LDR or shine light on it. The voltage will either increase with more light or decrease, depending on whether your LDR is the first or second resistor in the voltage divider circuit.
</details><hr>

---
# Project
### Build a Light-Sensitive Motor
Use a MOSFET transistor to control how much current is flowing through your motor. Gate the MOSFET with the output voltage of your light sensor...creating a motor that spins when the light is **ON** and stops when the light is **OFF**, or the other way around.
- *Help*: A guide to completing this task can be found here: [NB3-Building a Light-Sensitive Motor](https://vimeo.com/??????)
- *Hint*: Use the following circuit as a guide:
<p align="center">
<img src="../../../boxes/transistors/_resources/images/MOSFET_motor_driver.png" alt="MOSEFT driver" width="400" height="300">
</p>
