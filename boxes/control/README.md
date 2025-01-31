# The Last Black Box : Control
In this box, you will learn about control...

## Control
Getting a motor to move precisely how you want it to (direction and speed) is very important for *controlling* the behaviour of your robot. A number of very clever strategies have been developed to help you take **control** of your motor.

<details><summary><i>Materials</i></summary><p>

Name|Depth|Description| # |Data|Link|
:-------|:---:|:----------|:-:|:--:|:--:|
Servo Motor|01|FT90R Digital Micro Continuous Rotation Servo|2|[-D-](/boxes/control/)|[-L-](https://www.pololu.com/product/2817)
Servo Wheel|01|Wheels (70x8mm) for servos|2|[-D-](/boxes/control/)|[-L-](https://www.pololu.com/product/4925)
H-bridge|10|H-bridge motor driver (SN754410NE)|2|[-D-](/boxes/control/_resources/datasheets/sn754410.pdf)|[-L-](https://www.mouser.co.uk/ProductDetail/Texas-Instruments/SN754410NE)
DC Gearbox Motor|10|TT Gearbox DC Motor - 200RPM - 3 to 6VDC and wheel|2|[-D-](/boxes/control/)|[-L-](https://www.amazon.co.uk/AEDIKO-Motor-Gearbox-Shaft-200RPM/dp/B09V7QH1S8)

</p></details><hr>

#### Watch this video: [PWM](https://vimeo.com/1033905955)
> We can control a "continuous" range of outputs with a binary digital signal (only 0s and 1s) by switching the output **ON** and **OFF** very quickly. Our "continuous" output is then the average of the percentage of time spent **ON** vs **OFF**. We cal this percentage the "duty cycle", and we call this output control method *pulse width modulation* or **PWM**.


#### Watch this video: [Servo Loops](https://vimeo.com/1033963709)
> A servo loop connects feedback from a sensor to the control signals sent to a motor.


#### Watch this video: [H-Bridges](https://vimeo.com/1034209519)
> An H-Bridge allows sending current through a motor in both directions, and thus drive forwards *and* backwards.


# Project
### NB3 : Building a PWM Speed Controller
> We can use a digital (binary) signal and a MOSFET to turn a motor **ON** and **OFF**. We can use PWM to change the motor's speed.

<details><summary><weak>Guide</weak></summary>
:-:-: A video guide to completing this project can be viewed <a href="https://vimeo.com/1033891821" target="_blank" rel="noopener noreferrer">here</a>.
</details><hr>


