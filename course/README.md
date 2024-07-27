# The Last Black Box Course

> A kit containing a collection of black boxes that you "open" in sequence.

<p align="center">
<img src="_designs/layout/png/layout_LBB.png" alt="Course Layout" width="100%" height="100%">
</p>

----

## NB3

You will be building a robot. The robot's physical layout mimics the basic anatomy of a (vertebrate) brain. As you progress through the "boxes", your robot will *evolve* into an increasingly sophisticated machine. The goal of this course is to create an "intelligent" machine without using any "black boxes". We thus call this robot the No-Black-Box-Brain or NBBB or ***NB3***.

<p align="center">
<img src="_designs/NB3/NB3_render.png" alt="NB3 outline" width="300" height="300">
</p>

## Repository
The course is a combination of this code repo (LBB) and a submodule repo called "BlackBoxes". All of the subsequent exercises assume that you have "cloned" both repos to your Host computer. In much of the example code, there is an additional assumption that the LBB repo is in a folder called "NoBlackBoxes" in your home/user directory. You can setup everything (on Linux) with the following commands.

```bash
# ...from your "home" directory
mkdir NoBlackBoxes  # Create NoBlackBoxes directory
cd NoBlackBoxes     # Change to NoBlackBoxes directory

# Clone LBB repo with submodules
git clone --recurse-submodules https://github.com/NoBlackBoxes/LastBlackBox
```

## Versions
There are shorted versions of this course that cover a specific subset of the "black boxes" and focus on specific themes.

- [Bootcamp](bootcamp/README.md): A one-week intensive introduction to the essentials of modern technology and neuroscience.

- [Build A Brain](buildabrain/README.md): Designed for secondary age students (14+), this hands-on course shows you how recent advances in science and technology have made it possible for humans to create intelligent machines...by building you own. 

- [Own Phone](ownphone/README.md): ***-Under Development-*** Get your "smartphone license"! Demonstrate to your parents that you can take control of your digital tools...before they take control you.

## Black Boxes

1. **Atoms**
    - *Tech*: Introduction to atoms, sub-atomic particles, and forces.
    - *Brain*: From physics to chemistry to biology to brains.
    - *Exercises*: Draw your favourite atoms and molecules.

2. **Electrons**
    - *Tech*: Intro to basic electronics: voltage, current, resistance, batteries, Ohm's Law, voltage dividers, and power (dissipation).
    - *Brain*: Neurons (resting potential), passive properties.
    - *Exercises*: Measure voltage/current/resistance. Build a voltage divider. Turn on a light bulb.

3. **Magnets**
    - *Tech*: Intro to magnetism
    - *Brain*: Why not?
    - *Exercises*: Build a telegraph...coil an electromagnet

4. **Light**
    - *Tech*: Intro to EM radiation and spectrum
    - *Brain*: Why not?
    - *Exercises*: Lightbulbs and antennas

5. **Sensors**
    - *Tech*: Intro to transduction of heat, light, pressure, and sound.
    - *Brain*: Intro to photoreceptors, hair cells, and mechanosensors.
    - *Exercises*: Build a light, heat, pressure sensor using a photoresitor, thermistor, piezo

6. **Motors**
    - *Tech*: Intro to electromagnetism and piezos
    - *Brain*: Muscles and motor neruons (chemical synapses?)
    - *Exercises*: Wind a coil, spin a motor, make a sound, build a theremin(?)

7. **Transistors**
    - *Tech*: Intro to tubes and transistors
    - *Brain*: Action potentials and axons and synapses (or later...with decisions?)
    - *Exercises*: Switch on a motor with your sensor, better theremin?

8. **Amplifiers**
    - *Tech*: Intro op amps
    - *Brain*: Multiplicative NMJ, gain
    - *Exercises*: Move a motor with your sensor, better theremin?

9. **Circuits**
    - *Tech*: Intro to integrated circults; how are they made and what are they good for.
    - *Brain*: Simple sensorimotor behaviour
    - *Exercises*: Build a Braitenberg vehicle

10. **Power**
    - *Tech*: Voltage regulators
    - *Brain*: Efficiency and homeostasis
    - *Exercises*: Install NB3_power

11. **Data**
    - *Tech*: Getting from analog to digital (0 and 1s is all you need), ADCs and DACs
    - *Brain*: Neural code? (rate v timing?)
    - *Exercises*: Comparator...Build a simple ADC?

12. **Logic**
    - *Tech*: digital logic and the basis of computation
    - *Brain*: Simple neural circuits: E and I
    - *Exercises*: Build an adder

13. **Memory**
    - *Tech*: flip/flop, flash, storage
    - *Brain*: Synapses, LTP, and NMDA channels
    - *Exercises*: Sample hold circuit? (clapper?) Build a D-Latch

14. **FPGAs**
    - *Tech*: Programmble logic devices, HDL (verilog)
    - *Brain*: simple, adaptable circuits for computation
    - *Exercises*: Adder in FPGA...ALU...cpu

15. **Computers**
    - *Tech*: ALU, microcontrollers and progamming I
    - *Brain*: basic brains (brain computer anlogy debate)
    - *Exercises*: Arduino basics, blinky in ASM and C

16. **Control**
    - *Tech*: Negative feedback and servos, H-bridge, PID
    - *Brain*: Motor control
    - *Exercises*: Write direction, speed (and position?) controller

17. **Robotics**
    - *Tech*: Smarter robots
    - *Brain*: Smarter bot
    - *Exercises*: Ardunio based robot (PWM motors? various sensors?): Task: ?

18. **Systems**
    - *Tech*: Operating systems and programming II
    - *Brain*: Brain systems (sense, perceive, memory, learning, )
    - *Exercises*: Booting and connecting

19. **Linux**
    - *Tech*: Linux and command lines
    - *Brain*: Should the interfaces to your brain be "open"?
    - *Exercises*: Linux-life

20. **Python**
    - *Tech*: Programming in Python, installing packages, and virtual environments
    - *Brain*: How to think like a programmer?
    - *Exercises*: Talk to your hindbrain

21. **Networks**
    - *Tech*: Internet protocols and WiFi
    - *Brain*: Physical layer and neural protocols
    - *Exercises*: SSH and connect to bot via ESP or NRF

22. **Websites**
    - *Tech*: HTML, CSS
    - *Brain*: ??
    - *Exercises*: Build a nice looking website

23. **Servers**
    - *Tech*: HTTP requests and responses
    - *Brain*: ??
    - *Exercises*: Host your nice looking website

24. **Security**
    - *Tech*: Encryption
    - *Brain*: ??
    - *Exercises*: Mine a bitcoin?

25. **Audio**
    - *Tech*: From mics to "understanding" sound
    - *Brain*: Extracting information from hair cells, sound localization
    - *Exercises*: Build a sound localizer (dealing with 1D data)

26. **Vision**
    - *Tech*: From cameras to "vision"
    - *Brain*: Extracting information from photoreceptors (through V1 and beyond)
    - *Exercises*: Build a colored blob detector

27. **Learning**
    - *Tech*: Reinforcement learning and clicker training
    - *Brain*: RL in brains
    - *Exercises*: Clicker train yourself and your robot

28. **Intelligence**
    - *Tech*: Neural Networks and modern "AI", NPUs, LLMs
    - *Brain*: From fish to humans, evolution of biological intelligence
    - *Exercises*: NPU and tensorflow...mysteries...

----

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />The entire LastBlackBox repository and website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
