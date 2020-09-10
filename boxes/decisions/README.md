# Decisions

***SHOULD THIS BE COMBINED WITH DATA?***

thresholds -- as a switch
triggering events
nonlinear curves give you this switching / event
e.g. sound level detector -- level goes below some level
schmitt trigger!

----

<details><summary><b>Materials</b></summary><p>

Contents|Description| # |Data|Link|
:-------|:----------|:-:|:--:|:--:|
Comparator|LM339|2|[-D-](https://www.ti.com/lit/ds/symlink/lm339-n.pdf)|[-L-](https://uk.farnell.com/texas-instruments/lm339n/ic-comparator-quad-dip14-339/dp/3118457?st=lm339n)

Required|Description| # |Box|
:-------|:----------|:-:|:-:|
Multimeter|(Sealy MM18) pocket digital multimeter|1|[white](/boxes/white/README.md)|

</p></details>

----

## Positive Feedback

Feedback is what enables autonomy in computing devices. Once a mechanism is able to become "aware" of it's outputs and use them as inputs to start a computation anew, the system gains some sense of autonomy.

The Schmitt Trigger uses positive feedback to hold a value once an input has crossed a threshold. This is valuable because it's a kind of basic memory. This works based on the principle of hysteresis. This is a nonlinear phenomenon where as an input increases, the output only changes if a threshold input level is crossed. The inverse happens as the input crosses a lower threshold. In this way, the input can vary continuously, but the output is only ever in one of two states (on or off, True or False, 1 or 0, +V or -V).

We will build a Schmitt Trigger using a device called a **comparator**. Comparators are similar to (or might be described as) [operational amplifiers](https://www.wikiwand.com/en/Operational_amplifier), though they have specific properties which make them respond very fast (nanoseconds) to input. Op-amps are one of the most fundamental modern analog computing elements. They have many nuances, but in general they measure the difference in potential of their inputs and output an amplified (a value multiplied by some gain) version of this difference. This is why op-amps are useful in applications such as loudspeakers.

A comparator outputs a nonlinear version of this logic. The comparator takes two input voltages and *compares* them. So the comparator's output **C** at each level is:

<p align="center">
	<img src="https://render.githubusercontent.com/render/math?math=C_n=1\, \ \textrm{if}\ V_n<V_{input}">
</p>
<p align="center">
	<img src="https://render.githubusercontent.com/render/math?math=C_n=0\, \ \textrm{if}\ V_n>V_{input}">
</p>

If the input voltage is greater than the reference voltage (the other input), it goes "high" and else goes low.

With this logic, we can measure whether our input has crossed a threshold. In order to hold onto that value, we add feedback. If the output is high, we simply feed that output back into the input in order to "hold onto" the threshold crossing. Here's a schematic of one implementation:

### Exercise: Build a Schmitt-Trigger!

<p align="center">
	<img src="images/schmitt_trigger.png">
</p>

If you're really brave, forget the comparator black-box and build one out of trusty transistors! Be warned, however, they're trick [to get right](https://electronics.stackexchange.com/questions/164297/how-exactly-does-a-comparator-work).

<p align="center">
	<img src="images/schmitt_trigger_transistors.png">
</p>

### Squid Neurons

The [Schmitt Trigger](https://en.wikipedia.org/wiki/Schmitt_trigger) was developed based on Otto Schmitt's work on the squid neuron. Schmitt was attempting to build an "all-or-nothing" neuron model in hardware. The trigger might be called the first neuromorphic!

### References

[Applications for Schmitt Triggers](https://components101.com/articles/schmitt-trigger-introduction-working-applications)

[LM139](https://www.ti.com/product/LM139)

[SO Schmitt Trigger Help](https://electronics.stackexchange.com/questions/282502/how-to-build-a-schmitt-trigger-using-transistors)

----
