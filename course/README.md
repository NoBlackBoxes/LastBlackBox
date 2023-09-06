# The Last Black Box Course

> A kit containing a collection of black boxes that you "open" in sequence.

<p align="center">
<img src="designs/layout/layout.png" alt="Course Layout" width="500" height="375">
</p>

----

## NB3

You will be building a robot. The robot's physical layout mimics the basic anatomy of a (vertebrate) brain. As you progress through the "boxes", your robot will *evolve* into an increasingly sophisticated machine. The goal of this course is to create an "intelligent" machine without using any "black boxes". We thus call this robot the No-Black-Box-Bot or NBBB or ***NB3***.

## Repository
The course is a combination of this code repo (LBB) and a submodule repo called "BlackBoxes". All of the subsequent exercises assume that you have "cloned" both repos to your Host computer. On Linux, there is an additional assumption that the LBB repo is in a folder called "NoBlackBoxes" in your home directory. You can setup everything with the followinf commands.

```bash
# ...from your "home" directory
mkdir NoBlackBoxes  # Create NoBlackBoxes directory
cd NoBlackBoxes     # Change to NoBlackBoxes directory

# Clone LBB repo with submodules
git clone --recurse-submodules https://github.com/NoBlackBoxes/LastBlackBox
```

## Boxes

Hindbrain|Midbrain|Forebrain|
---------|--------|---------|
[1. Electrons](/boxes/electrons/README.md)|[9. Power](/boxes/power/README.md)|[17. Systems](/boxes/systems/README.md)
[2. Magnets](/boxes/magnets/README.md)|[10. Data](/boxes/data/README.md)|[18. Networks](/boxes/networks/README.md)
[3. Light](/boxes/light/README.md)|[11. Logic](/boxes/logic/README.md)|[19. Security](/boxes/security/README.md)
[4. Sensors](/boxes/sensors/README.md)|[12. Memory](/boxes/memory/README.md)|[20. Hearing](/boxes/hearing/README.md)
[5. Motors](/boxes/motors/README.md)|[13. FPGAs](/boxes/fpgas/README.md)|[21. Vision](/boxes/vision/README.md)
[6. Transistors](/boxes/transistors/README.md)|[14. Computers](/boxes/computers/README.md)|[22. Learning](/boxes/learning/README.md)
[7. Amplifiers](/boxes/amplifiers/README.md)|[15. Control](/boxes/control/README.md)|[23. Intelligence](/boxes/intelligence/README.md)
[8. **Reflexes**](/boxes/reflexes/README.md)|[16. **Behaviour**](/boxes/behaviour/README.md)|[24. **?**](/boxes/?/README.md)

## Timeline

### Week 1

- Day 1: Electrons + Magnets
- Day 2: Light
- Day 3: Sensors + Motors
- Day 4: Transistors + Amplifiers
- Day 5: Reflexes

### Week 2

- Day 1: Power, Data, Logic, Memory
- Day 2: FPGAs
- Day 3: Computers
- Day 4: Control
- Day 5: Behaviour

### Week 3

- Day 1: Systems, Networks, Security
- Day 2: Hearing, Vision
- Day 3: Learning, Intelligence?
- Day 4: *NB3 work*, *NB3 work*
- Day 5: *NB3 work*, *NB3 demos*

## Descriptions

1. **Electrons**

    - *Tech*: Intro to basic electronics: voltage, current, resistance, batteries, Ohm's Law, voltage dividers, and power (dissipation).
    - *Brain*: Neurons (resting potential), passive properties.
    - *Exercises*: Measure voltage/current/resistance. Build a voltage divider. Turn on a light bulb.

2. **Magnets**

    - *Tech*: Intro to magnetism
    - *Brain*: Why not?
    - *Exercises*: Build a telegraph...coil an electromagnet

3. **Light**

    - *Tech*: Intro to EM radiation and spectrum
    - *Brain*: Why not?
    - *Exercises*: Lightbulbs and antennas

4. **Sensors**

    - *Tech*: Intro to transduction of heat, light, pressure, and sound.
    - *Brain*: Intro to photoreceptors, hair cells, and mechanosensors.
    - *Exercises*: Build a light, heat, pressure sensor using a photoresitor, thermistor, piezo

5. **Motors**

    - *Tech*: Intro to electromagnetism and piezos
    - *Brain*: Muscles and motor neruons (chemical synapses?)
    - *Exercises*: Wind a coil, spin a motor, make a sound, build a theremin(?)

6. **Transistors**

    - *Tech*: Intro to tubes and transistors
    - *Brain*: Action potentials and axons and synapses (or later...with decisions?)
    - *Exercises*: Switch on a motor with your sensor, better theremin?

7. **Amplifiers**

    - *Tech*: Intro op amps
    - *Brain*: Multiplicative NMJ, gain
    - *Exercises*: Move a motor with your sensor, better theremin?

8. **Reflexes**

    - *Tech*: Intro to control
    - *Brain*: Simple sensorimotor behaviour
    - *Exercises*: Build a Braitenberg vehicle

9. **Power**

    - *Tech*: Voltage regulators
    - *Brain*: Efficiency and homeostasis
    - *Exercises*: Install NB3_power

10. **Data**

    - *Tech*: Getting from analog to digital (0 and 1s is all you need), ADCs and DACs
    - *Brain*: Neural code? (rate v timing?)
    - *Exercises*: Comparator...Build a simple ADC?

11. **Logic**

    - *Tech*: digital logic and the basis of computation
    - *Brain*: Simple neural circuits: E and I
    - *Exercises*: Build an adder

12. **Memory**

    - *Tech*: flip/flop, flash, storage
    - *Brain*: Synapses, LTP, and NMDA channels
    - *Exercises*: Sample hold circuit? (clapper?) Build a D-Latch

13. **FPGAs**

    - *Tech*: Programmble logic devices, HDL (verilog)
    - *Brain*: simple, adaptable circuits for computation
    - *Exercises*: Adder in FPGA...ALU...cpu

14. **Computers**

    - *Tech*: ALU, microcontrollers and progamming I
    - *Brain*: basic brains (brain computer anlogy debate)
    - *Exercises*: Arduino basics, blinky in ASM and C

15. **Control**

    - *Tech*: Negative feedback and servos, H-bridge, PID
    - *Brain*: Motor control
    - *Exercises*: Write direction, speed (and position?) controller

16. **Behaviour**

    - *Tech*: Smarter robots
    - *Brain*: Smarter bot
    - *Exercises*: Ardunio based robot (PWM motors? various sensors?): Task: ?

17. **Systems**

    - *Tech*: Operating systems and programming II
    - *Brain*: Brain systems (sense, perceive, memory, learning, )
    - *Exercises*: Python basics, linux basics

18. **Networks**

    - *Tech*: Internet protocols and WiFi
    - *Brain*: Physical layer and neural protocols
    - *Exercises*: SSH and connect to bot via ESP or NRF

19. **Security**

    - *Tech*: Encryption
    - *Brain*: ??
    - *Exercises*: Mine a bitcoin?

20. **Hearing**

    - *Tech*: From mics to "understanding" sound
    - *Brain*: Extracting information from hair cells, sound localization
    - *Exercises*: Build a sound localizer (dealing with 1D data)

21. **Vision**

    - *Tech*: From cameras to "vision"
    - *Brain*: Extracting information from photoreceptors (through V1 and beyond)
    - *Exercises*: Build a colored blob detector

22. **Learning**

    - *Tech*: Reinforcement learning and clicker training
    - *Brain*: RL in brains
    - *Exercises*: Clicker train yourself and your robot

23. **Intelligence**

    - *Tech*: Neural Networks and modern "AI", NPU
    - *Brain*: From fish to humans, evolution of biological intelligence
    - *Exercises*: NPU and tensorflow...mysteries...

24. ***The Last Black Box***
    - *Tech*: What are we missing?
    - *Brain*: What are we missing?
    - *Exercises*: Image a brain slice (golgi stain? Nissl?)

----

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />The entire LastBlackBox repository and website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
