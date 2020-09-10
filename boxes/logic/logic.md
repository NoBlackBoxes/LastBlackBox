# Digital Logic


### Analog Computing

<img src="images/water_integrator.png" width=300>
<img src="images/analog_analogues.png" width=300>

<img src="images/MONIAC_LSE.png" width=300>
<img src="images/MONIAC.png" width=300>

<img src="images/ACE.png" width=300>
<img src="images/colossus.png" width=300>
<img src="images/EDVAC.png" width=300>

<img src="images/ball_and_disk_integrator.png" width=300>
<img src="images/differential_analyzer.png" width=300>
<img src="images/difference_engine.png" width=300>
<img src="images/babbage.png" width=300>
<img src="images/ada_lovelace.png" width=300>

<img src="images/vlsi_image_sensor.png" width=300>







### Relays and Vacuum Tube Computing

<img src="images/Zuse.png" width=300>
<img src="images/Z2.png" width=300>

<img src="images/relay_circuit.png" width=300>

<img src="images/AlanTuring.png" width=300>
<img src="images/MaxNewman.png" width=300>

<img src="images/vacuum_tube_video.png" width=300>
<img src="images/relays.png" width=300>

<img src="images/ENIAC.png" width=300>

<img src="images/relay_schematic.png" width=300>

<img src="images/neumann.png" width=300>
<img src="images/neumann_architecture.png" width=300>

<img src="images/TommyFlowers.png" width=300>


### Transistor Computing

<img src="images/first_transistor.png" width=300>
<img src="images/first_transistor_schematic.png" width=300>

<img src="images/BC237.png" width=300>
<img src="images/transistor_schematic.png" width=300>
<img src="images/transistor.png" width=300>

<img src="images/spacewar.png" width=300>
<img src="images/PDP1.png" width=300>

<img src="images/ic_layout.png" width=300>
<img src="images/vlsi_book.png" width=300>

<img src="images/attiny_zoomin.png" width=300>
<img src="images/attiny_layout.png" width=300>

### VLSI

<img src="images/stereolithography.png" width=300>
<img src="images/stl_schematic.png" width=300>
<img src="images/moores_law.png" width=300>


### Logic Gates

<img src="images/or_gate.png" width=300>
<img src="images/logic_symbols.png" width=300>
<img src="images/and_gate.png" width=300>

<img src="images/gates.png" width=300>
<img src="images/gate_quiz.png" width=300>
<img src="images/cmos_gate_quiz.png" width=300>

<img src="images/diode_XNOR.png" width=300>


### Logical Computation

<img src="images/adders.png" width=300>
<img src="images/adder.png" width=300>

<img src="images/half_1bit_adder.png" width=300>
<img src="images/full_1bitadder.png" width=300>
<img src="images/3bit_adder.png" width=300>
<img src="images/2bit_adder.png" width=300>
<img src="images/3bit_adder2.png" width=300>



# (Very Basic) Intro to Digital Logic

Today we're working towards the engine of a processor This is working towards an ALU — there is much more to a CPU!

[Photo of CPU architecture]

What is "digital"?

Why do we need logic?

How do we implement logic in circuits?

What can we make with hardware logic?

- any instruction set can be represented as binary
- boolean algebra applied to such instructions / data can generate arbitrary computations with enough time and memory (see later discussions of interpretation/compilation)

Choices are made by people:

[https://en.wikipedia.org/wiki/Von_Neumann_architecture](https://en.wikipedia.org/wiki/Von_Neumann_architecture)

[https://en.wikipedia.org/wiki/Universal_Turing_machine](https://en.wikipedia.org/wiki/Universal_Turing_machine)

- Make an OR and AND gate
    - Explain how to get to XOR, NAND, etc
- Make a full adder out of XOR ICs
- Hook full adders together to get multiple bit adder?
- Explain negative numbers, subtraction, bit-shifting
- Link to reference of how to implement every single basic gate with NAND
-

Many ways to make XOR gate:

- [https://hackaday.io/project/8449-hackaday-ttlers/log/150147-bipolar-xor-gate-with-only-2-transistors/](https://hackaday.io/project/8449-hackaday-ttlers/log/150147-bipolar-xor-gate-with-only-2-transistors/)

IMPORTANT: make sure to pull up/down unused gates otherwise nondeterministic behavior may ensue.

NOTE: with BJTs you might see different currents on the outputs because the currents will add as you saturate transistors. Thus LEDs will glow differently depending on the circuit's state.

OR Gate

![](images/Untitled-1425ddf3-2a5f-4d93-819d-d496101f1a87.png)

AND Gate

Great example: [http://www.technologystudent.com/elec1/dig8.htm](http://www.technologystudent.com/elec1/dig8.htm)

![](images/Untitled-c9b50a7b-b105-4d02-9076-e82a15cc6148.png)

XOR Gate (using diode bridge + PNP)

I wasn't able to get this circuit working, but I didn't try very hard! PNPs flip my brain inside out... I think the issue was getting the incoming logic level to be reversed.

![](images/Untitled-825442e4-7951-450f-8e10-9dec8556a108.png)

![](images/Untitled-c64085f7-0edc-4b42-a1cc-5abe9fa621b3.png)

Gate Abstractions

![](images/Untitled-6c814281-f78d-4b8c-b474-133df32c7a46.png)

Half Adder

Parts

(XOR)[[https://www.ti.com/lit/ds/symlink/cd4030b-mil.pdf](https://www.ti.com/lit/ds/symlink/cd4030b-mil.pdf)]

(AND)[[https://www.ti.com/lit/ds/symlink/cd4081b.pdf](https://www.ti.com/lit/ds/symlink/cd4081b.pdf)]

(OR)[[http://www.ti.com/lit/ds/symlink/cd4071b.pdf](http://www.ti.com/lit/ds/symlink/cd4071b.pdf)]

Tutorial — can you spot the error?

[http://www.learningaboutelectronics.com/Articles/Half-adder-circuit.php](http://www.learningaboutelectronics.com/Articles/Half-adder-circuit.php)

![](images/Untitled-15dc1d1c-ae8d-4f1f-9de6-8ace11ec0102.png)

Full Adder

![](images/Untitled-ffdf0f39-5a96-4e09-82a3-97ac97632af7.png)

2-bit adder (without carry in)

![](images/Untitled-1f3de71c-9511-423e-8d3f-6279cdc74fb9.png)

3 bit adder

![](images/Untitled-0a55f581-5781-4604-a3c6-368bfc1b6b0b.png)

![](images/Untitled-3702979f-fc4e-4943-930a-c7a3d33d4887.png)

![](images/Untitled-fd2ff237-b9e8-4360-bd6f-e16c7945d05a.png)


## images

1822 — tabulate polynomials by finite differences (subtract and add to previous values) — in the Science Museum
Babbage's proposed Analytical Engine, considerably more ambitious than the Difference Engine, was to have been a general-purpose mechanical digital computer. The Analytical Engine was to have had a memory store and a central processing unit (or ‘mill’) and would have been able to select from among alternative actions consequent upon the outcome of its previous actions (a facility nowadays known as conditional branching). The behaviour of the Analytical Engine would have been controlled by a program of instructions contained on punched cards connected together with ribbons (an idea that Babbage had adopted from the Jacquard weaving loom). Babbage emphasised the generality of the Analytical Engine, saying ‘the conditions which enable a finite machine to make calculations of unlimited extent are fulfilled in the Analytical Engine’ (Babbage [1994], p. 97).

1886
adding big numbers together was hard!
Consider an example system that measures the total amount of water flowing through a sluice: A float is attached to the input carriage so the bearing moves up and down with the level of the water. As the water level rises, the bearing is pushed farther from the center of the input disk, increasing the output's rotation rate. By counting the total number of turns of the output shaft (for example, with an odometer-type device), and multiplying by the cross-sectional area of the sluice, the total amount of water flowing past the meter can be determined.
https://en.wikipedia.org/wiki/Ball-and-disk_integrator
Lord Kelvin and his brother James Thomson

Penn 1942 — Differential analyser — Vannevar Bush
“A differential analyser may be conceptualised as a collection of ‘black boxes’ connected together in such a way as to allow considerable feedback. Each box performs a fundamental process, for example addition, multiplication of a variable by a constant, and integration. In setting up the machine for a given task, boxes are connected together so that the desired set of fundamental processes is executed. In the case of electrical machines, this was done typically by plugging wires into sockets on a patch panel (computing machines whose function is determined in this way are referred to as ‘program-controlled’).”

https://en.wikipedia.org/wiki/Water_integrator — 1936
for specific problems and general modular system
The water level in various chambers (with precision to fractions of a millimeter) represented stored numbers, and the rate of flow between them represented mathematical operations. This machine was capable of solving inhomogeneous differential equations.
https://history-computer.com/CalculatingTools/AnalogComputers/Lukianov.html.
Vladimir Lukianov — solving PDEs for temperature distributions / heat equations in concrete (for railway design) — changed temperature to water pressure and flow rate
In 1941, Lukyanov created a hydraulic integrator of modular design, which made it possible to assemble a machine for solving various problems. Two-dimensional and three-dimensional hydraulic integrators were designed.

analog analogues -- this is from a 1953 technical report from US army / MIT Civil Eng about analog computing

<p align="center">
	<img src="images/MONIAC.png" width="400">
	<img src="images/MONIAC_LSE.png" width="400">
</p>

LSE 1949 — inputs and outputs for a simulated economy
monetary national income analogue computer
~2% precision with digital/analytical techniques


<p align="center">
<img src="images/relay_schematic.png" width="400" >
<img src="images/relay_circuit.png" width="400">
<img src="images/relays.png" width="300">
</p>

electro(mechanical) switch
ARRA, Harvard Mark II, Zuse Z2, and Zuse Z3
literally an electromagnetic switching device

<p align="center">
	<img src="images/Zuse.png" height="300">
	<img src="images/Z2.png" height="300">
</p>

The Z2 was a mechanical and relay computer completed by Konrad Zuse in 1940. It was an improvement on the Z1, using the same mechanical memory but replacing the arithmetic and control logic with electrical relay circuits.

<p align="center">
	<img src="images/TommyFlowers.png" height="300">
	<img src="images/ENIAC.png" height="300">
</p>

It was the development of high-speed digital techniques using vacuum tubes that made the modern computer possible.
Tommy Flowers — British Post Office, used for telephone exchanges
triodes as switches — heat the cathode, gives of e-’s, change the voltage on the gate (bias) to change the current that can flow through to the anode.
The 1946 ENIAC computer used 17,468 vacuum tubes and consumed 150 kW of power
led directly to the building Colossus (Turing's Enigma breaking machine)


#### Vacuum Tubes



cathode heats up, throws off electrons
anode positively charged, current flows

<p align="center">
	WATCH THIS VIDEO WITH SOUND!!! <br>
	<a href="https://www.youtube.com/watch?v=hwutHPYGgfU">
		<img src="images/vacuum_tube_video.png" width="600">
	</a>
</p>

### Colossus (1940s)

Colossus — 1600 thyratron/thermionic tube — 1943-45

Colossus lacked two important features of modern computers. First, it had no internally stored programs. To set it up for a new task, the operator had to alter the machine's physical wiring, using plugs and switches. Second, Colossus was not a general-purpose machine, being designed for a specific cryptanalytic task involving counting and Boolean operations.

150kW


EDVAC — Electronic Discrete Variable Automatic Computer (binary)
used for missile trajectories
The ENIAC (1946) was the first machine that was both electronic and general purpose. It was Turing complete, with conditional branching, and programmable to solve a wide range of problems, but its program was held in the state of switches in patchcords, not in memory, and it could take several days to reprogram.

ACE Computer — Manchester
Max Newman established the Royal Society Computing Machine Laboratory at the University of Manchester, which produced the world's first working, electronic stored-program electronic computer in 1948, the Manchester Baby. Also Bletchley park colossus research
Turing's design had much in common with today's RISC architectures and it called for a high-speed memory of roughly the same capacity as an early Macintosh computer (enormous by the standards of his day).
With an operating speed of 1 MHz, the Pilot Model ACE was for some time the fastest computer in the world.

> …all the essential ideas of the general-purpose calculating machines now being made are to be found in Babbage's plans for his analytical engine. In modern times the idea of a universal calculating machine was independently introduced by Turing … [T]he machines now being made in America and in this country … [are] in certain general respects … all similar. There is provision for storing numbers, say in the scale of 2, so that each number appears as a row of, say, forty 0's and 1's in certain places or "houses" in the machine. … Certain of these numbers, or "words" are read, one after another, as orders. In one possible type of machine an order consists of four numbers, for example 11, 13, 27, 4. The number 4 signifies "add", and when control shifts to this word the "houses" H11 and H13 will be connected to the adder as inputs, and H27 as output. The numbers stored in H11 and H13 pass through the adder, are added, and the sum is passed on to H27. The control then shifts to the next order. (Newman, 1948)


### Von Neumann (1950s)

John realized that “Data” and a “Program” do not require distinct substrates…they can be stored in the same hardware, system memory (often RAM) and then processed by a separate CPU.

https://en.wikipedia.org/wiki/Von_Neumann_architecture#Von_Neumann_bottleneck
comes from turing’s work on stored program concept

The original Harvard architecture computer, the Harvard Mark I, employed entirely separate memory systems to store instructions and data. The CPU fetched the next instruction and loaded or stored data simultaneously and independently. This is in contrast to a von Neumann architecture computer, in which both instructions and data are stored in the same memory system and (without the complexity of a CPU cache) must be accessed in turn. The physical separation of instruction and data memory is sometimes held to be the distinguishing feature of modern Harvard architecture computers. With microcontrollers (entire computer systems integrated onto single chips), the use of different memory technologies for instructions (e.g. flash memory) and data (typically read/write memory) in von Neumann machines is becoming popular. The true distinction of a Harvard machine is that instruction and data memory occupy different address spaces. In other words, a memory address does not uniquely identify a storage location (as it does in a von Neumann machine); it is also necessary to know the memory space (instruction or data) to which the address belongs.

Modern functional programming and object-oriented programming are much less geared towards "pushing vast numbers of words back and forth" than earlier languages like FORTRAN were, but internally, that is still what computers spend much of their time doing, even highly parallel supercomputers.
As of 1996, a database benchmark study found that three out of four CPU cycles were spent waiting for memory. Researchers expect that increasing the number of simultaneous instruction streams with multithreading or single-chip multiprocessing will make this bottleneck even worse.[28] In the context of multi-core processors, additional overhead is required to maintain cache coherence between processors and threads.


pdp-1
spacewar invented by steve russell

### Transistors (1960s)


### Very Large-Scale Integration (1970s)

1978 carver mead lynn conway = VLSI