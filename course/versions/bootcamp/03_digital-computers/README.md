# Bootcamp : Digital Computers
It's time to learn how transistors can perform logic, store information, and work together in a circuit that *computes*.

## Logic
Boole was coole.

<details><summary><i>Materials</i></summary><p>

Name|Depth|Description| # |Data|Link|
:-------|:---:|:----------|:-:|:--:|:--:|

</p></details><hr>

#### Watch this video: [Logic Gates](https://vimeo.com/1033231995)
> The essential elements of computation (NOT, AND, OR, XOR, etc.) can be built from straight forward combinations of MOSFETs.


## Memory
There are many ways to store information.

<details><summary><i>Materials</i></summary><p>

Name|Depth|Description| # |Data|Link|
:-------|:---:|:----------|:-:|:--:|:--:|

</p></details><hr>

#### Watch this video: [Flash Memory](https://vimeo.com/1033230293)
> Storing much of your data requires *quantum mechanics*.


## Computers
It may not yet seem believable, but you can build a **computer** by combining transistors in a clever way. **Let's learn how!**

<details><summary><i>Materials</i></summary><p>

Name|Depth|Description| # |Data|Link|
:-------|:---:|:----------|:-:|:--:|:--:|
Microcontroller|01|Arduino Nano (rev.3)|1|[-D-](/boxes/computers/_resources/datasheets/arduino_nano_rev3.pdf)|[-L-](https://uk.farnell.com/arduino/a000005/arduino-nano-evaluation-board/dp/1848691)
Piezo Buzzer|01|Piezoelectric speaker/transducer|1|[-D-](/boxes/computers/_resources/datasheets/piezo_buzzer.pdf)|[-L-](https://uk.farnell.com/tdk/ps1240p02bt/piezoelectric-buzzer-4khz-70dba/dp/3267212)
Cable (MiniUSB-1m)|01|Mini-USB to Type-A cable (1 m)|1|[-D-](/boxes/computers/)|[-L-](https://uk.farnell.com/molex/88732-8602/usb-cable-2-0-plug-plug-1m/dp/1221071)

</p></details><hr>

#### Watch this video: [Architecture](https://vimeo.com/1033601146)
> The basic building blocks of a computer (memory, ALU, clock, bus, and IO) have a standard arrangement (architecture) in modern systems.


#### Watch this video: [NB3 : Hindbrain](https://vimeo.com/1033609727)
> We will now add a *computer* to our robot. We be using a simple microcontroller as our NB3's hindbrain. It will be responsible for controlling the "muscles" (motors) in response to commands from another (larger) computer that we will be adding later to the NB3's midbrain.

**TASK**: Mount and power your Arduino-based hindbrain (connect the mini-USB cable)
<details><summary><strong>Target</strong></summary>
    The built-in LED on the board should be blinking at 1 Hz.
</details><hr>


### Low-Level Programming
> We can control a computer by loading a list of instructions ("operations") into its memory. This is called *programming*.


# Project
### NB3 : Building a Theremin
> Building a light-to-sound feedback loop musical instrument (theremin) using an Arduino, an LDR, and a Piezo buzzer.

<details><summary><weak>Guide</weak></summary>
:-:-: A video guide to completing this project can be viewed <a href="https://vimeo.com/1033896646" target="_blank" rel="noopener noreferrer">here</a>.
</details><hr>

**TASK**: Build a Theremin
- *Hint*: What if you used the analog voltage signal measured from your light sensor to change the frequency of the "tone" playing on your buzzer? Hmm...
<details><summary><strong>Target</strong></summary>
    You should here a sound that varies with your hand motion (in front of a light)
</details><hr>

**TASK**: ***Have fun!*** (Make something cool)
- This diagram of the Arduino "pins" will definitely be useful: ![Arduino Pinout](/boxes/computers/_resources/images/pinout_arduino_nano.png)
<details><summary><strong>Target</strong></summary>
    You should have fun!
</details><hr>


