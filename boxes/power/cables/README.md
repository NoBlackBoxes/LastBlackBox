# power : cables
Powering your NB3 requires cables to carry current to the relevant components. The performance of these cables starts to matter when the amount of current being carried increases.

## Jumper Cables
Many of our NB3's connections are made with "jumper cables". These cables contain thin strands of copper wire twisted together at the core and a surrounding insulator. The ends of these wires are standard "DuPont" (aka CRHO) connectors (plug or sockets), with a spacing of 2.54 mm (0.1") between neighboring connectors (in a multi-pin version), which is the same spacing as the solderless breadboard sockets and the RPi's GPIO header.

**Problem**: These jumper wires often use as little copper wire as possible (28 AWG, diameter: 0.321 mm, area: 0.0804 mm^2). This means they have a non-negligible resistance of ~ 0.212 Ohms per meter. However, including resistance at the contacts themselves, this could be up 1 Ohm, or 0.1 Ohms for the short 10 cm cables we use most often. 

When powering the NB3's motor and midbrain computer, we may occasionally need a few Amps of current to pass through these wires. This can cause problems. The wire will heat up (P = V^2/R) and the voltage will drop across the cable (V = IR). For 2 Amps, we might see a 0.2 Volt drop from our power supply to our RPi. This can (and often does) cause under-voltage warnings.

**Solution**: Use shorter, thicker cables and, when possible, more of them. This is why we connect four cables (2 x +5V and 2 x Ground) to our RPi's power pins. We also provide a specific "power jumper cable" for this connection, which is made with thicker, shorter wires (22 AWG, 5 cm).

(Photo of new cable here)

The cable between the NB3 power supply uses 24 AWG, 3 cam wires.

(Photo of NB3 power cable here)
