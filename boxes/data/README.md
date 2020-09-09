# Data

When we say "data", what are we really referring to? In this section, we'll start with what it means to be a number, and work up to understanding why our computers use a binary number system.

----

<details><summary><b>Materials</b></summary><p>

Contents|Description| # |Data|Link|
:-------|:----------|:-:|:--:|:--:|
Resistor Ladder|8 Resistor divider/ladder|1|-|[-L-](
)
Comparator|Comparator|8|-|[-L-](
)

Required|Description| # |Box|
:-------|:----------|:-:|:-:|
Multimeter|(Sealy MM18) pocket digital multimeter|1|[white](/boxes/white/README.md)|

</p></details>

----

## ADC

How do we go from an analog world to a binary number?

### Exercise: Build a simple ADC

To build an analog-to-digital converter (ADC), we'll use the simplest design called a "flash" ADC or a "parallel" ADC. The basic idea is to use a chain of voltage dividers to divvy the incoming analog signal into voltages ranges. The binary output signal then indicates which of these ranges your input signal covers. There are other designs for ADCs that are more accurate, but this one is simple and fast.

Here is a circuit diagram of the device:

<p align="center">
 <img src="ADC1.png" alt="Flash ADC" width="500" text-align="center">
</p>

The question now is, what are the arrows? Those are [comparators](https://www.wikiwand.com/en/Comparator), which are themselves composed of transistors. Comparators are similar to (or might be described as) [op-amps](https://www.wikiwand.com/en/Operational_amplifier), though they have specific properties which make them respond very fast (nanoseconds) to input.

The comparator-based ADC is, in one sense, effectively an analog computing element bridging the divide between analog computation and the world of 1s and 0s. The ADC takes in an analog signals and outputs a digital version of them.

Our goal here is to build the ADC shown in the schematic above to make a basic signal visualizer. The ADC will use a reference voltage (from the power supply) divided into voltage steps by a line of resistors. Each of these steps the voltage will be a fraction of the reference voltage.

<!-- HACK to get latex in here. See: https://gist.github.com/a-rodin/fef3f543412d6e1ec5b6cf55bf197d7b -->
<p align="center">
	<img src="https://render.githubusercontent.com/render/math?math=V_n = n\times\frac{V_{\textrm{reference}}}{\textrm{number\ of\ resistors}}">
</p>

For each voltage level, the comparator takes the input voltage and *compares* it to it's other input, the reference voltage at that level. So the comparator's output **C** at each level is like this:

<p align="center">
	<img src="https://render.githubusercontent.com/render/math?math=C_n=1\, \ \textrm{if}\ V_n<V_{input}">
</p>
<p align="center">
	<img src="https://render.githubusercontent.com/render/math?math=C_n=0\, \ \textrm{if}\ V_n>V_{input}">
</p>

So if the input voltage is greater than the current step, the **sign** function returns a 1, else it returns a 0. The comparator is performing *one-bit quantization*. With that, we're in a binary world.

The number of levels (number of resistor-comparator pairs) defines the resolution of the ADC. In the schematic above, we have a 2-bit ADC, as there are 4 possible levels. Thus we need 2 bits to encode each of the possible levels.

<p align="center">
	<img src="https://render.githubusercontent.com/render/math?math=\textrm{number\ of\ levels} = 2^{\textrm{number\ of\ bits}}">
</p>

The comparator is still a pretty magical black box, and it's doing something nonlinear as we can see from the equation for **C**. As it turns out, [comparators are quite complex](https://electronics.stackexchange.com/questions/164297/how-exactly-does-a-comparator-work). If you get your ADC working, have a go trying to make one from transistors.

----
