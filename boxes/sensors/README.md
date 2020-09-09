# sensors

Computers and brains work with voltages. In order for either to understand signals in the environment (light, sound, pressure, heat, etc.), then these signals must be converted into a voltage. This conversion is called *transduction* and the thing that does it is a *transducer*. However, given their role in sensing the environment, it is common to call these transduction devices ***sensors***.

----

<details><summary><b>Materials</b></summary><p>

Contents|Description| # |Data|Link|
:-------|:----------|:-:|:--:|:--:|
Thermistor|Temperature sensitive resistor|2|[-D-](_data/datasheets/thermistor.pdf)|[-L-](https://uk.farnell.com/epcos/b57891m0103k000/thermistor-ntc-radial-leaded/dp/2285471)
Photoresistor|Light-dependent resistor|2|[-D-](_data/datasheets/ldr.pdf)|[-L-](https://uk.farnell.com/advanced-photonix/nsl-19m51/light-dependent-resistor-550nm/dp/3168335)
Piezo|Piezo element|1|[-D-](_data/datasheets/piezo.pdf)|[-L-](https://uk.farnell.com/multicomp/mcabt-455-rc/audio-element-piezo-2-8khz-35mm/dp/2433035?st=piezo)

Required|Description| # |Box|
:-------|:----------|:-:|:-:|
Multimeter|(Sealy MM18) pocket digital multimeter|1|[white](/boxes/white/README.md)|
Test Lead|Alligator clip to 0.64 mm pin (20 cm)|2|[white](/boxes/white/README.md)|
Batteries (AA)|AA 1.5 V alkaline battery|4|[electrons](/boxes/electrons/README.md)|
Battery holder|4xAA battery holder with ON-OFF switch|1|[electrons](/boxes/electrons/README.md)|
Jumper kit|Kit of multi-length 22 AWG breadboard jumpers|1|[electrons](/boxes/electrons/README.md)|
Jumper wires|Assorted 22 AWG jumper wire leads (male/female)|1|[electrons](/boxes/electrons/README.md)|

</p></details>

----

## Measuring Heat

Heat is the average kinetic energy within a material. When materials change their temperature, their resiatnce to electric current also changes, and for some materials this change can be quite large. We can use these materials to make a "thermistor", a temperature dependent resistor.

<p align="center">
<img src="_images/thermistor.png" alt="Thermistor" width="150" height="150">
<p>

We can convert a thermistor's change in resistance to a change in voltage using a voltage divider. By replacing one of the divider's resistors with a thermistor, the voltage in the middle of the divider will now change with temperature. We will have built a temperature sensor that can talk to a computer.

<p align="center">
<img src="_images/heat_sensor.png" alt="Heat Sensor" width="150" height="150">
<p>

Your body senses temperature differently. Special proteins, called TRP (pronounced "trip") channels, straddle the insulating membrane surrounding a neuron. The neuron is more negative on the inside than outside, and thus positve ions outside would love to enter the cell. The TRP channels contain a small hole that is usually closed. However, when the protein reaches a specific temperature it changes its structure and the hole opens, allowing positive ions to flow into the cell and chnage its. This is how the nervous system converts temperature to voltage.

<p align="center">
<img src="_images/trp_channels.png" alt="TRP Channels" width="150" height="150">
<p>

There a different types TRP channels sensitive to different temperatures, some for hot and very hot, and others for cold and very cold. There is an interesting consequence of relying on specific channels for detecting specific temperatures. The TRP channels are proteins with part of their structure sticking out of the cell and exposed to whatever happens to be around. Some chemicals are able to bind to this exposed bit of the protein and change its structure, causing it to open. If this happens, then your brain will have no way of distinguishing this kind of opening from when the channel encounters a real change in temperature.

A famous chemical that binds to the TRP channel that reports "very hot" temperatures is called capsacin, and it is found in spicy foods. (Yes, the "heat" of spicy foods is an illusion of your body's temperature sensing system). Another famous chemical is menthol, which binds to "cold" sensing TRP channels. Another, slightly less well known chemical, is mustard oil, which binds to the "painfully cold" TRP channels. Mustard oil is found foods like wasabi and horseradish, and this "extreme cold illusion" explains the particular, and intense, sensory response.

### Exercise: Build a heat sensor

- Use your thermistor and a resistor (size of your choice) to build a voltage divider. Apply a voltage and monitor the output at the divider's midpoint.

***Q:*** *How does the choice of resistor influence the response of your heat sensor?*

***Q:*** *Should the termistor replace R1 or R2 in the voltage divider? What difference does this make for your sensor output?*

***Q:*** *How does the voltage output from your sensor relate to standard units of temperature (degrees C or F)? How could you "calibrate" your sensor?*

### Challenge: Convince your brain that spicy food is an illusion

- Get a very hot chilli pepper. Eat it. Tell your brain it is just an illusion. If your brain doesn't believe you. Use your heat sensor to measure the temperature of your mouth before and after eating the pepper.

***Q:*** Does your brain believe you now?

----

## Measuring Light

When light shines on some kinds of materials, the incoming photons can "bump" electrons out of their bonds and allow them the move freely throughout the material, decreasing its resistance. When the light is shut off, these liberated electrons are bound again and the resistance increases. Materials with high light sensitivity can be used to create light dependent resistors (also called photoresistors), which can be used to create a light sensor.

<p align="center">
<img src="_images/photoresistor.png" alt="Photoresistor" width="150" height="150">
<p>

Importantly, as long as the photons in the incoming light have enough energy to free electrons from their bonds, then they will change the resistance of the material. Therefore, this change in resistance tells us only about the amount of total light and not about the energy of the individual photons it contains (i.e. the colour of the light). In order to measure colour, we will need to tweak our sensor design, and we will get some clues from your eyes.

Your eye contains special cells called photoreceptors. These cells contain moulecules (called pigments) stuck in its membrane that interact well with incoming light. When light hits a pigment, it changes the shape of a small moleule (called retinal) attached to the pigment on the inside of the cell. This modification of retinal then triggers a cascade of chemical consequences inside the cell, ultimately resulting in the *closing* of small holes in the cell membrane and preventing the positive ion Na+ from entering. By preventing Na+ from entering, light makes the inside of a photoreceptor become more negative. This is how you convert changes inlight into changes in voltage that your brain can use "to see".

<p align="center">
<img src="_images/photoreceptors.png" alt="Photoreceptors" width="150" height="150">
<p>

Different photoreceptors make different pigment proteins, and each pigment is specifically sensitve to photons of particular energies (i.e. colours). You have pigments sensitive to red, green, and blue photons, but some animals make pigments sensitve to ultra-violet photons. If we want to build a *colour* sensor, then we need to make our light dependent resistor senstive to a specific range of colors. We could do this by selecting materials with specific colour sensitivty, such as the different pigments in your eye, or we could use a material sensitive to a large range of photons and then specifically block (with a filter) the colours we don't want. This is how computer light sensors (such as cameras) detect colour. Importantly, in order to correctly measure and identify all the same colours that our brains can see, we only need 3 different types of light sensors, one for red, one for green, and one for blue.

### Exercise: Build a light sensor

- Convert your heat sensor to a light sensor by replacing the thermistor with a photoresistor.

----

## Measuring Touch

There are materials that change their resistance when some external force is applied to them, and they can easily be used to build a force sensitve resistor following a similar design as above.

Your body actually detects touch in a similar way. There a cells found in your skin that have channels across the membran, when some force distorts the membrane, these channels open. In other words, pressure on the cell changes the resistance of the membrane, and thus the voltage of the cell.

<p align="center">
<img src="_images/mechanoreceptors.png" alt="Mechanoreceptors" width="150" height="150">
<p>

When the force is released the material must return to its previous shape in order to return to its previous resistance. This might take some time. A touch sensor built with such materials might be good at detecting when touch begins, but may be slow at detecting when it ends. However, there are other ways to measure force on a material, and some are better suited to measuring very fast changes.

A crystal is just a material with its atoms arranged in a regular pattern. Some crystals have the fascinating property that when they are squeezed, the electrons rearrange themselves such that more accumulate on one side of the crystal. We call such crystals pizeo-electric, because they can convert changes in force directly into changes in voltage.

<p align="center">
<img src="_images/piezo.png" alt="Piezo" width="150" height="150">
<p>

### Exercise: Build a touch sensor using a force sensing resistor

- Use the voltage divider design

### Exercise: Detct touch with a piezo

- Directly connect your pizeo to the probe terminals of your multimeter and measure voltage when applying pressure.

***Q:*** What happens when you apply constant pressure?

----

## Measuring Sound

Sound is caused by vibrations in the environment sending out waves of pressure changes through the air. These pressure changes travel quickly (i.e. the pseed of sound: ?) and contain rich information about what is going on the environment. We can measure these changes in air pressure using a similar process to how we measured direct pressure applied to a material.

<p align="center">
<img src="_images/piezo_mic.png" alt="Piezo Microphone" width="150" height="150">
<p>

These changes in air pressure are normally much smaller than the forces resulting from a phsyical touch. We will need to use a very sensitive sensor to detect them reliably.

Your ear contains special cells that have a tiny hair protruding for its membrane and into a fluid filled canal. The fluid is held in place by a thin membrane that is exposed directly to the air on the other side. Pressure changes in the air, sound, vibrate this thin drum, which then changes propogates these vibrations to the fluid on the other side. The hair of each cell is then pushed back and forth by the vibrations in the fluid. AT the base of the hair, tiny holes will open if the hair swings far enough away from its center position. When these holes open, positive ions flow quickly into the negative cell, increasing its voltage. This is how you convert sound to voltage.

<p align="center">
<img src="_images/hair_cells.png" alt="Hair Cells" width="150" height="150">
<p>

The hair on each hair cell is off different lengths, and this has an important consequence. Longer hairs will be moved more by slow vibrations, needing time to fully displace from their neutral position. Shorter hairs will be moved more by faster vibrations, as they are shorter and can respond more quickly to the quick changes. These various hair lengths mean that each hair cell will be sensitive to different frequencies of vibration. This is how you measure the frequency (or tone) of sounds arriving at your ear. (Talk about phase detection!!...and why this si possible...and what it gives you)...localization. *Oooh...could build a sound source loaclizer with two mics?*

### Exercise: Build a sound sensor (i.e. microphone)

- Attach your piezo mic element to the probe terminals of your multimeter.
- Hard to do this without an amplifier...