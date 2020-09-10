# Logic

Boole was cool.

----

<details><summary><b>Materials</b></summary><p>

Contents|Description| # |Data|Link|
:-------|:----------|:-:|:--:|:--:|
Gate (AND)|4xAND gate|1|-|-
Gate (OR)|4xOR gate|1|-|-
Gate (NOT)|4xNOT gate|1|-|-

Required|Description| # |Box|
:-------|:----------|:-:|:-:|
Multimeter|(Sealy MM18) pocket digital multimeter|1|[white](/boxes/white/README.md)|

</p></details>

----

# Digital Logic

The first step in making a modern digital computer is to understand how to perform logical (mathematical) operations on binary data. This means working towards an Arithmetic Logic Unit (ALU). As we will see, there is much more to a CPU, but this is the place to begin.

## Logic Gates

Familiarize yourself with these!

<p align="center">
<img src="images/gates.png" width=900>
</p>

Try making some gates from transistors:

<p align="center">
	<figure>
	<img src="images/or_gate.png" width=250>
	<img src="images/and_gate.png" width=250>
	</figure>
</p>

The gates above are described in terms of BJTs, can you figure out what gates are represented below using FETs?

<p align="center">
<img src="images/cmos_gate_quiz.png" width=800>
</p>

## Logical Computation

Draw the truth table for the following device. What does it do?

<p align="center">
	<figure>
		<img src="images/half_1bit_adder.png" height=150>
		<img src="images/full_1bitadder.png" height=150>
		<br><figcaption>
			1-bit half adder (left) and full adder (right).
		</figcaption>
	</figure>
</p>

It's an adder! Which also means it's a subtracter (with some additional logic).
<p align="center">
<img src="images/adders.png" width=900>
</p>

# Exercise: Build a 3-bit digital adder!

Build this! Use LEDs to visualize the output of your computation.

<p align="center">
	<img src="images/3bit_adder.png">
</p>


# Further Reading

[Can you spot the error in this tutorial?](http://www.learningaboutelectronics.com/Articles/Half-adder-circuit.php)

[There are many ways to make XOR gate](https://hackaday.io/project/8449-hackaday-ttlers/log/150147-bipolar-xor-gate-with-only-2-transistors/)



----
