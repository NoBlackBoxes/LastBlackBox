# Build a Brain : Session 4 - How Computers Work
Logic + Memory = Computer

## Power
Running more capable software requires a faster computer, which requires more power. We will first explore how a power supply works and then install one on your NB3.

#### Watch this video: [NB3-Power](https://vimeo.com/1005162740)
> This video

- [ ] **TASK**: Measure the voltage of 4xAA batteries in series (end to end).
- *Hint*: Use your battery holder.
<details><summary><strong>Target</strong></summary>
Batteries connected in series will sum their voltages. You should measure four times the voltage of a single AA battery, about 6.4 Volts, from the batteries in your 4xAA holder.
</details><hr>

- [ ] **TASK**: Add a (regulated) 5 volt power supply to your robot, which you can use while debugging to save your AA batteries and to provide enough power for the Raspberry Pi computer.
<details><summary><strong>Target</strong></summary>
Your NB3 should now look like this:
<p align="center">
<img src="../../../boxes/power/_data/images/NB3_power_wiring.png" alt="NB3 power wiring" width="400" height="300">
</p>
</details><hr>

---

# Project
### Build a Light-Sensitive Motor
Use a MOSFET transistor to control how much current is flowing through your motor. Gate the MOSFET with the output voltage of your light sensor...creating a motor that spins when the light is **ON** and stops when the light is **OFF**, or the other way around.
- *Hint*: Use the following circuit as a guide:
<p align="center">
<img src="../../../boxes/transistors/_data/images/MOSFET_motor_driver.png" alt="MOSEFT driver" width="400" height="300">
</p>
