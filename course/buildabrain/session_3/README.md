# Build a Brain : Session 3 - Digital decisions
The building blocks of our digital world.

## Transistors
The most important invention of the past 100 years. We have made more transistors than any other object...by far. Understanding how transistors work will provide the foundation for understanding all of the amazing devices we have built with them.

#### Watch this video: [Semiconductors](https://vimeo.com/1000842810)
> Stuff

#### Watch this video: [Diodes](https://vimeo.com/1000861996)
>
- [ ] **Task**: Illuminate a light-emitting diode (LED). *Remember the current limiting resistor*

#### Watch this video: [Transistors - MOSFETs](https://vimeo.com/1000873279)
>
- [ ] **Task**: Measure the threshold voltage that opens your MOSFET gate. Compare it to the "expected" range listed in the 
- The datasheet for your MOSFET can be found here: [IRF510](../../../boxes/transistors/_data/datasheets/IRF510.pdf)
<details><summary><strong>Target</strong></summary>
The threshold for when current starts to flow through your MOSFET ("Gate-Source Threshold Voltage") should be between 2 to 4 Volts. However, the amount of current it allows will rise rapidly up to (and beyond) 10 Volts. Check the datasheet (Figure 3). 
</details><hr>

---

# Project
### Build a Light-Sensitive Motor
Use a MOSFET transistor to control how much current is flowing through your motor. Gate the MOSFET with the output voltage of your light sensor...creating a motor that spins when the light is **ON** and stops when the light is **OFF**, or the other way around.
- *Hint*: Use the following circuit as a guide:
<p align="center">
<img src="../../../boxes/transistors/_data/images/MOSFET_motor_driver.png" alt="MOSEFT driver" width="400" height="300">
</p>
