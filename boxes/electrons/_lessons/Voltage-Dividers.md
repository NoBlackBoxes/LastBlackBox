# Electrons: Voltage Dividers
Controlling the level of voltage at different places in a circuit is critical to designing electronic devices.

## [Video](https://vimeo.com/1030787469)

## Concepts
- What happens to the voltage and current in different parts of your circuit?
- The current flowing through a simple circuit, without parallel paths, must be the *same* everywhere. Otherwise electrons would start to pile up.
- The voltage, the electric pressure, drops from +6V at the top to 0V at the bottom. Where is this electric pressure used up? dissipated?
- It is dissipated as "heat" in the resistor. The electrons deliver their energy to the material, increasing the motion of the atoms in the resistor, heating it up. Thus the potential energy of the electrons in the battery is converted to kinetic energy of atoms in the resistor.
- If use other devices, then we can convert this potential energy into other forms of energy, motion in a motor, or light in an LED.
- Now let's take a look at a slightly more complicated circuit with two resistors.
- The current through both resistors is the same.
- The voltage throughout the circuit, however, must change. It starts as +6V and ends at 0V. Where does the voltage change? Where is the electric pressure dissipated? In the resistors.
- Electric pressure is lost in each resistor! If the resistors are the same value, then the amount they dissipate will be the same.
- If we measure how much electric pressure, or voltage, remains after just one resistor, then we expect it to be half of what we started with.
- What if the resistors are different sizes? Intuitively, we expect the larger resistor to dissipate more pressure...and our intuition is correct.
- We can also work out exactly what will happen.
- If R1 is 220 Ohms and R2 is 470 Ohms, then we have a total resistance of 690 Ohms.
- The current is given by Ohm's Law. 6.4V / 690 = ~ 9 mA
- So how much does the voltage drop in R1. This can also be determined by Ohm's Law. 220 * 0.009 = ~2 Volts. Thus, we would expect to measure the remaining 4.4 Volts after R1.
- These circuits are called Voltage Dividers. We can determine Vout as Vin * (R2/ (R1 + R2))
- They allow us to produce any voltage level we need by choosing resistors of different values.
- One convenient way you will find voltage dividers is as a potentiometer. A device with three connections and this underlying circuit. (draw circuit)
- We can vary the position of pin 2. We can change the ratio of R1 and R2, and thus we can change the output voltage on pin 2.
- Potentiometers are often found underneath the knobs you turn on electronic devices...for example, when you change the volume on the stereo.

## Connections

## Lesson

- **TASK**:Build a voltage divider using two resistors of the same value. Measure the intermediate voltage (between the resistors).
> With equal size resistors, the intermediate voltage you measure should be half of the supply voltage.

- **TASK**:Build a voltage divider using a variable resistor (potentiometer). Measure the intermediate voltage. What happens when you change the position of the internal contact of the variable resistor (by turning the screw)?
  - *Help*: A video guide to completing these tasks can be found here: [NB3-Building Voltage Dividers](https://vimeo.com/1000789632)
> The intermediate voltage should vary continuously as you adjust the potentiometer.
