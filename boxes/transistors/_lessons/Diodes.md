# Transistors: Diodes
The chemical and electrical equilibrium between charge carriers creates a potential across the PN junction. This junction potential only permits current flow in one direction, which gives **diodes** there delightfully non-linear behavior.

## [Video](https://vimeo.com/1032443724)

## Concepts

## Connections
- When we place P and N type piece of Silicon next to one another, something very interesting occurs.
- The charge carriers, electrons and holes, diffuse from where there are many to where there are fewer.
- However, when they leave, the electrons leave behind a net positive charge...and the holes a net negative. This voltage works against the diffusion until a balance is reached.
- We call the voltage in this equilibrium state the Junction Potential.
- The JP depends on the concentration of dopants.
- Now, what happen when we try to pass a current through such a device sandwich (PN junction)?
- If we put lots of electrons, a negative voltage on the P side, then we are just working in the same direction of the JP...extending it. There is still no flow through the junction.
- If we put a positive charge on the P-ytype, then we are working against the JP...an when we reach a voltage greater than JP, electrons can move across the gap. Current can flow.
- After this threshold, then the harder we push, the more current flows.
- The IV curve is non-linear, which gives these devices, Diodes, their rectifying property.
- Non-linear relationships such as this are fundamental in building more interesting circuits, in particular those that can perform computation. 
- It also looks very similar to ReLU, the activation function most commonly used in artificial neural networks.
- In summary, diodes are devices that only conduct current in one direction (up to a point).
- They have an internal voltage that must be over come before current starts to flow.
- In LEDs, pushing electrons across the Junction causes them to accelerate when they "snap" into a hoe on the other side. This emits light, a photon, where the color is based on the properties of the material....the size of the band gap.

## Lesson

- **Task**: Illuminate a light-emitting diode (LED). *Remember the current limiting resistor!*
> The LED should only illuminate when installed in one orientation. If you flip it around, then the "diode" of the LED will prevent current flowing through the circuit.
