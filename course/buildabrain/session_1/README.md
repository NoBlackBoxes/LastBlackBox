# Build a Brain : Session 1 - Sensors
In this session we will learn how electricity can be used to create "sensors" (inputs for our robot brain). Sesnors can measure anything we want we want our robot to know about the world (light, temperature, pressure, sound, etc.). We will open the following "black boxes" during this session:
- Atoms (01), Electrons (01), Sensors (01)

## Atoms
> "Let's start at the very beginning, a very good place to start". - *R&H*

- *Watch this video*: [Structure and the Periodic Table](https://vimeo.com/1000458082)
<img src="../../../boxes/electrons/_data/images/dipole_field_template.png" alt="dipole field template" height="250" style="border: 2px solid #000000;"/>
  - When you need it **(and you will)**, then you can find the periodic table [here](../../../boxes/atoms/card/periodic_table.png)

## Electrons
- *Watch this video*: [Voltage](https://vimeo.com/1000730032)
  - **Task**(measure_voltage_AA): Measure the voltage of a AA battery using your Multimeter (select voltage ("V") and touch your probes to either end of the battery).[number]
  - **Task**(measure_voltage_4xAA): Measure the voltage of 4xAA batteries in series (end to end). *Hint*: You can use your battery holder.[number]
- *Watch this video*: [Conductors](https://vimeo.com/1000740989)
- *Watch this video*: [Current](https://vimeo.com/1000743561)
- *Watch this video*: [Resistors](https://vimeo.com/1000755493)
  - **Task**(measure_resistors): Measure the resistance of one of your resistors.[number]

With a voltage source (battery) and resistors, then we can start building "circuits" - complete paths of conduction that allow current to flow from a location with fewer electrons (+) to a location with more electrons (-).
> ***Note***: This is *weird*. Electrons are the things moving. Shouldn't we say that current "flows" from the (-) area to the (+) area? Unfortunately, current was described before anyone knew about electrons and we are stuck with the following awkward convention: **Current is defined to flow from (+) to (-)**...even though we now know that electrons are moving the opposite way.

To help you build electronic circuits, we will assemble a "prototyping platform", which also happens to be the body of your robot (NB3).

- *Watch this video*: [NB3 Body](https://vimeo.com/1005036900)
  - **Task**(assmeble_NB3_body): Assemble the robot body (prototyping base board). Upload a photo of your NB3.[photo]
    - If you are curious how the *NB3 Body* printed circuit board (PCB) was designed, then you can find the KiCAD files here: [NB3 Body PCB](../../../boxes/electrons/NB3_body)

- *Watch this video*: [Ohm's Law](https://vimeo.com/1000768334)
  - **Task**(measure_current): Build the simple circuit below and measure the current flowing when you connect the battery pack.[number] You know the voltage from the batteries (V) and the resistance of the resistor (R). Does Ohm's Law hold?[text]
    - ***Note***: Measuring current with a multimeter is ***tricky***. You can only get an accurate measurement if ***ALL*** of the current in the circuit is forced to flow through your multimeter. This means that when measuring current, your multimeter must be in *series* with the rest of the circuit. (As opposed to measuring voltage, when your multimeter is placed "parallel" to the circuit.)

- *Watch this video*: [Voltage Dividers](https://vimeo.com/1000782478)
  - **Task**(build_a_voltage_divider): Build a voltage divider using two resistors of the same value? Measure the intermediate voltage (between the resistors).[number]
  - **Task**(build_a_variable_voltage_divider): Build a voltage divider using a variable resistor (potentiometer). Measure the intermediate voltage. What happens when you change the position of the internal contact (by turning the screw)?[text]
    - A video guide to completing these tasks can be found here: [Building Voltage Dividers](https://vimeo.com/1000789632)

## Sensors
- *Watch this video*: [Light Sensors](https://vimeo.com/1000794164)

---

# Project
### Build a Light Sensor
- Build a voltage divider with one of the fixed resistors replaced with an LDR (light-dependent resistor). Does the "output" voltage vary with light level? What should the value of the fixed resistor be to maximize the sensitive range of the output voltage for varying light levels?
- A guide to completing this task (and all the morning circuit building tasks) can be found here: [Building Circuits](https://vimeo.com/1005054579)
