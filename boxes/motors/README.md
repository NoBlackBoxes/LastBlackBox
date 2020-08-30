# Motors

Computers and brains work with voltages. In order for either to affect the environment, then voltage must converted into something that can act upon the world (light, sound, heat, motion, etc.). This conversion from voltage to action is called *actuation* and a thing that does it is an *actuator*. Given that we are building a robot, we will place particular emphasis on actuators that produce movement, ***motors***.

----

<details><summary><b>Materials</b></summary><p>

Contents|Description| # |Data|Link|
:-------|:----------|:-:|:--:|:--:|
DC Motor|Brished DC motor|1|-|-
Whistle|Motor shaft attachment|1|-|-

Required|Description| # |Box|
:-------|:----------|:-:|:-:|
Multimeter|(Sealy MM18) pocket digital multimeter|1|[white](/boxes/white/README.md)|

</p></details>

----

## Electric motors

Electrons flowing through a conductor induce a magentic field circling the conductor. By coiling the the conductor, we can align these magentic fields to create a stronger, current-dependent "electromagnet". We can thus use voltage to control current to control a magentic field to move a magent, and some devices need just this to convert voltage into useful movements.

<p align="center">
<img src="_images/telegraph.png" alt="Telegraph" width="150" height="150">
<p>

The simple telgraph was just a electromagnetic coil placed beneathe a permenent magent attached to a pring-loaded hammer. When a pulse of current arrived in the coil, perhaps from a distant telegraph office, then the coil would energize, attract the magnet, and produce an audible click. Pretty straightforward, but world-changingly useful.

The magnetic force that an electromagnet exerts can be large, but it decays quickly with distance. Therefore, the movements a coil can actuate are normally quite small. In order to produce more "useful" movements with voltages (that cause currents in coils), we need to get creative.

<p align="center">
<img src="_images/electric_motor.png" alt="Electric Motor" width="150" height="150">
<p>

By placing a coil next to a permanent magent attached to a rotating rod, energising the coil will cause the magnet to try an align itself with the new magnetic field, rotating the rod until its North pole points to the coil South pole. If we to control this motion with more precision, then we can arrange a circle of coils around the central magnet and activate them in sequence to systematically rotate the magent around. The rate at which we switch between the active coils wil determine the speed the rod rotates, and the order will determine the direction. This is an electric motor.

There are many ways to control how we activate the correct sequence of coils, but one of the most straightforward (and convenient) is to have the rotation of the rod, automatically "connect" the next coil. This is accomplished by dragging a brush of conductive wires along with the rotating magnet and rod. The brush has metal bristles that connect the next coil in the circuit, thus continuing the rotation, advancing the brush to the next coil, and so on.

Brushed DC (DC because the use a stable current) motors are simple, but they have some drawbacks. First, there is no way to tell the motor to stop at particular position, as the brush will always reach ahead to activate the next coil. Second, the brush dragging creates friction (ahich causes it to wear out oevr time) and noise (which is just annoying). However, the cheap and simple design will be useful for building a simple robot and you will get used their whining over time.

If we want to control the position of the motor, rather than just its speed, then we will still use a similar design for the physical motor, but we will take more control over how the coils are activated. However, in order to take more control over activating the correct coils, in the correct sequence, at the correct times, we are going to need some more advanced electronic devices. These will come in later boxes.

### Exercise: Move your motor

- Spin your motor. Connect the leads (red and black wires) to the + and - terminals of your battery pack.

***Q:*** *What happens when you switch the + and - connections?*

- Roate your motor by hand. You are rotating a magnet (and the brush) across the unpowered electromagnetic coils. Now connect the leads of your motor to each other (just twist them together).

***Q:*** *Does it feel different to roate the motor (magnet) with the leads connected? If so, why?*

- You should can control the speed of your motor by controling how much current flows through the coils. You can do this my varying the voltage you apply or by applying fixed voltage and varying a resistor in series with the motor. Try to use your LDR to create a light-controlled-motor!

***Q:*** *Did it work? Why or why not?*

----

## Muscles

Your muscles work differently.

<p align="center">
<img src="_images/muscles.png" alt="Muscles" width="150" height="150">
<p>

### (literal) Exercise: Push-ups

- Do 20 push-ups (it is Boot-Camp after all). Wonder at the power of your myosin.

----

## Speakers

Turning voltage into sound is quite similar to creating movement.

<p align="center">
<img src="_images/speaker.png" alt="Speaker" width="150" height="150">
<p>

### Exercise: Build a piezo buzzer

- Build some sensible oscillator (?) and power a buzzer or homemade speaker...

----
