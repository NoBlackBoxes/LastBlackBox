# Electrons

Electrons are sub-atomic particles with a negative electric charge. Their negative charge produces a *force* that attracts positive charges and repels negative charges.

<p align="center">
<img src="_images/electron.png" alt="Electron" width="200" height="200">
<p>

In this box, we will learn about electricity...how to measure it and how to **control** it.

----

<details><summary><b>Materials</b></summary><p>

Contents|Description| # |Data|Link|
:-------|:----------|:-:|:--:|:--:|
Resistor|330 &Omega;/0.25 W|2|[-D-](_data/datsheets/resistor.pdf)|[-L-](https://uk.farnell.com/multicomp/mf25-330r/res-330r-1-250mw-axial-metal-film/dp/9341730)
Resistor|470 &Omega;/0.25 W|1|[-D-](_data/datsheets/resistor.pdf)|[-L-](https://uk.farnell.com/multicomp/mf25-470r/res-470r-1-250mw-axial-metal-film/dp/9341943)
Resistor|1 k&Omega;/0.25 W|1|[-D-](_data/datsheets/resistor.pdf)|[-L-](https://uk.farnell.com/multicomp/mf25-1k/res-1k-1-250mw-axial-metal-film/dp/9341102)
Resistor|10 k&Omega;/0.25 W|2|[-D-](_data/datsheets/resistor.pdf)|[-L-](https://uk.farnell.com/multicomp/mf25-10k/res-10k-1-250mw-axial-metal-film/dp/9341110)
Capacitor|0.1 uF ceramic capacitor|1|[-D-](_data/datasheets/capacitor_ceramic.pdf)|[-L-](https://uk.farnell.com/kemet/c322c104k1r5ta/cap-0-1-f-100v-10-x7r/dp/1457685)
Capacitor|100 uF aluminum electrolytic capacitor|1|[-D-](_data/datsheets/capacitor_electrolytic.pdf)|[-L-](https://uk.farnell.com/rubycon/16zlh100mefc5x11/cap-100-f-16v-20/dp/8126283)
Button|Tactile switch|2|[-D-](_data/datasheets/button.pdf)|[-L-](https://uk.farnell.com/omron/b3f-1000/switch-spno-0-05a-24v-tht-0-98n/dp/176432)
Potentiometer|20 k&Omega; variable resistor|2|[-D-](_data/datasheets/pot_20kOhm.pdf)|[-L-](https://uk.farnell.com/bourns/3362p-1-203lf/trimmer-20k/dp/9354344)
Breadboard (400)|400-tie solderless breadboard|1|[-D-](_data/datasheets/breadboard_400.pdf)|[-L-](https://uk.farnell.com/multicomp/mcbb400/breadboard-solderless-300v-abs/dp/2395961)
Balloon|Any color-latex|1|-|-
Batteries (AA)|AA 1.5 V alkaline battery|4|-|-
Battery holder|4xAA battery holder with ON-OFF switch|1|-|[-L-](https://www.dfrobot.com/product-202.html)
Battery case|4xAA battery storgae case|1|-|[-L-](https://www.amazon.co.uk/gp/product/B07ZHZDWQ2)
Jumper kit|Kit of multi-length 22 AWG breadboard jumpers|1|[-D-](_data/datasheets/jumper_kit.pdf)|[-L-](https://uk.farnell.com/multicomp/mc001810/hard-jumper-wire-22awg-140pc/dp/2770338)
Jumper wires|Assorted 22 AWG jumper wire leads (male/female)|1|[-D-](_data/datasheets/jumper_wires.pdf)|[-L-](https://uk.farnell.com/multicomp/mcbbj65/jumper-wire-assortment-65pcs/dp/2396146)
Rubber feet|Adhesive rubber standoffs (1421T6CL)|4|[-D-](_data/datasheets/rubber_feet.pdf)|[-L-](https://uk.farnell.com/hammond/1421t6cl/feet-stick-on-pk24/dp/1876522)

Required|Description| # |Box|
:-------|:----------|:-:|:-:|
Multimeter|(Sealy MM18) pocket digital multimeter|1|[white](/boxes/white/README.md)|
Test Lead|Alligator clip to 0.64 mm pin (20 cm)|2|[white](/boxes/white/README.md)|
Body|Laser cut base (5 mm clear acrylic)|1|[reflexes](/boxes/reflexes/README.md)|

</p></details>

----

## Atoms

Atoms are the tiny particles that make up all (normal) matter. They are composed of three types of sub-atomic particles: protons, neutrons, and electrons. The protons and neutrons are clustered together in a central core, called the nucleus, while the electrons "orbit" around the nucleus in rings.

<p align="center">
<img src="_images/atom.png" alt="Atom" width="200" height="200">
<p>

The mass of an atom is determined by the number of protons and neutrons in its nucleus (the more the heavier), while the behaviour of the atom (what it reacts with/sticks to/etc.) is determined by the arrangement of electrons in the rings (called an "orbit"). The orbit closest to the nucleus is the smallest and can only contain two electrons. The next two orbits are larger and can each contain up to 8 electrons. The fourth orbit can contain 18. The electron capacity of each orbit is determined by quantum mechanics, but the consequences are readily observable.

<p align="center">
<img src="_images/valence.png" alt="Valence" width="200" height="200">
<p>

The outermost orbiting electrons are called the "valence" electrons, as these are the ones that can interact with other atoms. If the outermost orbit is full (has all slots filled with electrons), then this atom will not want to react with other atoms. We call such materials "inert" (e.g. Helium and Argon). However, if the outermost orbital is not full, the this atom will be quite interested in interacting with other atoms willing to share one of their valence electrons, particularly those that have a complemetary number of valence electrons. For example, an atom with just 1 electron in its third orbital (Sodium, Na) is quite interested in an atom with 7 electrons in its third orbital (Chloride, Cl). Sodium and Chloride, when they find each other, can share valence electrons with each other, completing their outer rings. When two atoms share electrons, we call this a "bond", as they are now stuck together. The more electrons they share, the stronger the bond.

## Electric field

The attractive and repulsive forces surrounding each charge can be described as a field. This *electric field* just says what would happen to a "test charge" placed at any location nearby. The fields created by all of the individual charges add together to create a more complex field. The simplest example of a combined field is with one negative and one positve charge, which is called a "dipole" because it has two poles, one postive and one negative.

<p align="center">
<img src="_images/electric_field.png" alt="Electric Field" width="150" height="150">
<p>

In many materials, the balance of positive and negative charges will cancel and the electric field is very weak. However, it is possible to produce an imbalance of charge between two regions and create a strong electric field between them, which we can use to push around electrons.

### Exercise: Static electricity

- Materials can exchange electrons when in contact with one another. Some materials (called "triboelectric") have a tendency to lose or gain electrons when they are seperated. If a "lossy" material repeatedly makes contact (i.e. is rubbed against) a "gainy" material, then a net charge will accumulate. This excess charge creates a (quite strong) electric field that you can use to impress small children.
  - Inflate your balloon
  - Rub it against your shirt (best if made of wool) or hair. The balloon will acquire an excess negative charge as it tends to hold onto electrons when seperated from hair-like materials that have weakly affiliated electrons.
  - This negatively chargeed balloon can be used to do all sorts of things...experiment.
    - One thing you should defintely *not do* is touch a charged ballon to an electronic device. Although you cannot be hurt by the small number of electrons that the balloon has accumulated, they tiny elements inside an electornic device (of which we will learn much more about) can eaily damaged.

----

## Voltage

When there is a difference in the density of electrons between two regions, then there will be a force that wants to push the negative electrons towards the more positive (less-negative) region, creating a potential for electrcity to flow. This difference in concentration is therefore called a "potential difference", or Voltage.

![Voltage](images/voltage.png)

Voltage is measured in Volts (abbreviated V) and you will interact with voltages of various strengths. Your wall socket provides a voltage of arround 220 V (in Europe and the UK) or 110 V (in the USA). A typical electronic device will use voltages of 5 V or 3.3 V. Your brain produces voltages of ~100 millivolts (mV, where 1 mV is 1/1000<sup>th</sup> of a Volt).

Voltages can be created by any process that produces an unequal concentration of positive and negative charges. Naturally, seperating charges requires doing some work and a source of energy is required to create a voltage.

Batteries use chemical energy to seperate charges onto its two sides, with one side more positive and one side more negative. The common AA alkaline battery is a "cell" containing a conductive solution with electrolytes. A reaction between a metal "electrode" placed in the solution exhanges charges between the electrode and the solution, and charges accumulate on the electrode. At the same, the opposite electrode has charges pulled from its surface into the solution. This chemical reaction creates an excess of negative charge on one electrode and an absence of negative charge (effectively a net positive charge) on the other. The reaction between electrode and solution continues until the repulsive forces caused by the accumulated charge exactly balance the force of the chemical reaction. For a typical alkaline battery, this balance (equilibrium) is reached around 1.5 V.

![Battery](images/battery.png)

What happens when a battery runs out? How can it be recharged?

There are many different chemistries that can be used to create battery cells. One of the most popular uses Lithium ions. These cells have a voltage of 3.7 V and they can store a lot of charge.

The cells of your brain, Neurons, are also batteries. They use special proteins called pumps that use the energy in an ATP moluecule to exchange an ion of Sodium for an ion of Potassium.

![Resting Potential](images/resting_potential.png)

### Exercise: Measure voltage of a battery

- Use your multimeter to measure the voltage of a AA alakline battery. A multimeter can measure many different (hence *multi*) properties of electricity and electronic componenets.
  - Select voltage mode (V)
  - Select range (10V)
  - Place the red (+) and black (-) probes on the ends of the battery.

- Q: Why is the voltage not exactly 1.5 Volts?

----

## Conductors, Insulators, and Resistance

If we manage to produce a voltage between two regions, then we still need a path between these regions for electrons to travel, otherwise not much will happen.

Electrons are normally tightly bound to their atomic nucleus (which contains positively charged protons and neutral neutrons). However, for some atoms, the outermost shell of electrons is very weakly bound and these outer electrons are free to move from one atom to another. These materials are *conductors*, because they can conduct a flow of electrons...electricity.

![Conductors](images/conductors.png)

Many conductors are metals, but not all. Electricity can also be conducted through liquids that have positive and negative ions called electrolyes dissolved in them. This is why water (with a bit of salt) can also conduct electrcity. Your brain uses water containing the electrolyes Sodium (Na+), Potassium (K+), Chloride (Cl-), and Calcium (Ca2+) (and a few others) to conduct electricity.

![Electrolyes](images/electrolytes.png)

Not all materials (metals or solutions) conduct electricty equally well. If a metal is mixed with other atoms that don't have free outer electrons, then the metal atoms' free electrons will have a more difficult time moving around. In liquid conductors, the amount of dissolved electrolytes will determine how easily electricty can flow. We quantify how easily electricity flows through a material by measuring its conductance (in Siemens, abbreviated S). Inversely, and more commonly, we can quantify how *hard* it is for electricity to flow by measuring its resitance (in Ohms, abbreviated &Omega;). Resistance is the inverse of conductance: &Omega; = 1 / S.

We can engineer mixtures of materials with precisely the value of resistance we want. These devices are called resistors and we will use them often to control the flow of electricty.

![Resistors](images/resistors.png)

Some materials have a very hard time conducting electricity. They are either solids with tightly bound electrons or liquids without any dissolved electrolytes. They therefore have a very high resitance and impede the flow of electricity. We call these materials insulators, becuase they are often used to prevent the flow of electricity to places we don't want it to go. The plastic coatings surrounding a metal wire is an insulator.

Your brain uses fat as an insulator, a thin layer of lipids that surround the conductive solution of electrolytes and direct the flow electricity to where it is needed. These lipid insulators define the shapes of the electrical units in your brain, cells called Neurons.

![Insulators](images/insulators.png)

Values for the resitances of different materials (in different shapes and sizes) are listed below:

TO DO: Table fo resistances of wires, glass, water, neural membranes

### Exercise: Measure the reistance of different resistors

- You can measure resistance using a multimeter. Here we will use its ability to measure resitance in Ohms.
  - Select resitance mode (&Omega;)
  - Select a range (100 to 10k)
  - Touch the red and black metal probe tips to either end of a resistor.
  - Wait for the value on the screen to stablise.

- Measure each resistor in your box. The resistors have colored bands that indicate their value. It is an arcane skill, but if you'd like to learn to identify resistors by the bands, then you might save yourself a bit of time with the multimeter (and acquire a rare geek badge).

----

## Current

When a conductive material connects twos regions with different concentrations of electrons (i.e. with a "potential difference" or voltage between them), then charge will flow between these regions through the conductor. This flow of charge is called an electric current and is quantified as Amperes (charges flowing per second, abbreviated A).

If you could look very closely at ends of the conductor, you would see electrons leaving the negative electrode and entering the conductor. At the other end, you would see charges leaving the conductor and entering the positve electrode. Just to be clear, the same electron that enters the negative side is not the same electron the leaves the positive side. The free (outer) electrons of the metal conductor are all simply pushed along by the electric field of the arriving new electrons, and the ones on the end are pushed out. This allows currents to propogate very, very fast, faster than possible if we had to wait for an electron to move through the entire conductor. For good conductors, this speed approaches the speed of light.

![Current](images/current.png)

In solutions, things are similar. In the nervous system...

![Axon](images/axon.png)

We use a number of currents in building electronic devices. Many are very small (microamps, 1 millionth of an Amp), but some will be larger (approaching an Amp). These larger currents are needed to make stuff happen in the world, like lighting up a lamp or moving a small motor. Fortunately, currents that only need to convey information from one place to another can be quite small. For example, the currents required to make an "old school" phone call over a wire are ~15 milliamps (mA). The currents that flow into and out of a neuron are on the order of picoamps (1 millionith of a microamp).

You will often see references to two different kinds of currents: direct current (DC) and alternating current (AC). These are both also referring to the voltages that produce the currents. A DC voltage, such as that produced by a battery, is stable and will produce a constant current in one direction. An AC voltage, like that available from your home's electrical socket, is variable. It fluctuates between positive to negative voltage levels and will produce currents the move back and forth within a conductor at 50 (in Europe and the UK) or 60 (in the USA) time per second. Computers work with DC voltages and we will have to convert from AC to DC voltages to power our computer with a plug from the wall. This is what those black cubes do found in many a phone charger.

### Exercise: Measure current across a resistor

- Use your multimeter to measure the current flowing through a resistor.
  - Select current mode (A)
  - Select range (1A)
  - Build a simple test circuit (need to explain circuit!)
  - Place the red (+) and black (-) probes on the ends of the battery.

- Q: What is the relationship between the resistor you use and the current flowing? Measure the current flowing through three resistors and make a graph.

----

## Ohm's Law

How much current flows when a voltage is applied across a material depends on that material's resistance. The relationship between voltage, current, and resitance is given by Ohm's Law. It is a simple linear equation that states that the current flow is equal to the voltage divided by the resistance.

<p align="center">
<img src="_images/ohms_law.png" alt="Ohm's Law" width="300" height="150">
</p>

Ohm's Law is very useful. Look at extremes...low R, high V?

### Exercise: Measure current flowing across 470 Ohm resistor

----

## Capcitors

Although electrons can only move through conductors, their electric fields can extend beyond the material (even through a vacuum). We can use this fact to produce some interesting, non-Ohm's law electronic devices, which will be very useful in taking control of electricty.

If two conductors are placed very near one another, but not quite in contact, then the electric field created on one conductor can influence the other. If a voltage is placed across this conductive gap, then charge will accumulate on one side, creating an electric field that will repel charge on the other side. In other words, pushing electrons onto the first conductor, will end up pushing electrons out of the second conductor. If we simple ignored the gap and only looked at the ends of the wire, then we would see a current flowing as if there was just a straightforward connection in between...at least initially.

<p align="center">
<img src="_images/capacitor.png" alt="Capacitor" width="300" height="150">
</p>

Devices such as this, with two conductors in close proximity but not in contact (arranged such the electric field on one influences the other) are called capacitors, because they have the capacity to store charge, i.e. capacitance. Capcitance is measured in Farads (abbreviated, F).

Capacitors do not obey something like Ohm's law, in which applying a constant voltage will produce a constant current. Instead, when a voltage is first applied to a capacitor, current can flow very easily (as if R was low), however, as charge accumulates on the capacitor, it gets harder and harder to push more charge, and it appears as f the resistance of the device increases. This increase in apparent resistance is exponential, it starts low (essentially zero) in increases to infinity as the capacitor charges. How quickly does it increase? This depends on how quickly we put charge on the capacitor, which depends on the current.

<p align="center">
<img src="_images/chargin_capacitor.png" alt="Charging Capacitor" width="300" height="150">
</p>

Capacitors provide a time-dependence to our electornic devices. They will help us to create and manipulate signals that vary in time, an obvioulsy useful ability for anything operating in a dynamic world.

There is another way to influence the timing of electricty, but that will require exploiting its deep and fundamental connection to magnetism.

### Exercise: Measure current and voltage across a capacitor when connected to a battery

## Circuits

The essential elements of electrcity (voltage, current, resistance, and capcitance) can be combined to create all sorts of fascinating and useful devices...including machines that think. We can combine them into a network of interconnected elements, called a circuit, and in the following exercises we will build some simple examples.

### Exercise: Simple circuit with switch and meter (Test bench)...introduce breadboard)

----

## Volatge Didviders

### Exercise: Build a 50% voltage divider

### Exercise: Build a variable voltage divider

----

## Power

Power (abbreviated P) is a measure of energy per unit time. When energy is quanitfied in Joules (abbreviated J) and time is quantified in seconds (abbreviated s), then power is quanitfied in Watts (abbreviated W). The power generated in a resistor with current flowing through it is given by the following equation: P = V<sup>2</sup> / R. Ohm's Law allows us to rewrite this as P = VI. This makes sense. The voltage tells us how hard we are pushing on an electron and current tells us how many electrons are being moved per second.

A power supply, a device used to provide power (voltage and current) to an electronic circuit, is usually rated in Watts (W). A resistor also comes with rating for how much power it "can handle" before breaking. A typical resistor is rated to tolerate 0.25 W.

Resistors dissapate power in two ways, as heat and as light.

### Exercise: Power in circuits

- How much power must be dissapated when you connect a 470 &Omega; resistor to a 1.5 V battery? What about a 3 &Omega; resistor?
- Can a 15 W power supply spin a 12 V motor that requires at leat 1.5 A of current?

## Heat

All materials are made of tiny particles that are constantly in motion. This is true for gases and liquids, but also for solids, where the particles might not move very far but are always shaking. This random motion of particles in a material is called heat. The average kinetic energy of these random motions is a material's temperature, which is quantified in degrees (Celsius, abbreviated C, or Farhenheit, abbreviated F).

When current flows through a resistor, the moving electrons interact with the material and increase the movement of its constiuent particles, increasing its temperature. The more current that flows, the more heat is generated. Sometimes this is useful, such as when we push a lot of current through a coil of wire in an electric tea kettle to quickly heat up some water. Sometimes this is not useful, such as when too much current flows in one of electronic devices, causing it to heat up too much and break. We can do our best to prevent this by calculating the "power" that will be generated and then check whether our device can handle (i.e. dissapte) that amount of power safely.
