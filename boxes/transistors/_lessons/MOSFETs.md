# Transistors : MOSFETs
MOSFETs are the thing that humans have built more of than anything else. They must be useful! Let's discuss what they are and how they work.

## [Video](https://vimeo.com/1032452466)

## Concepts
- The world had a problem. It needed to take the tiny signals on a thin conductor and send it across the entire country. The further these signals traveled, the weaker they got.
- The only ways we new how to regenerate these signals often failed (vacuum tubes).
- How can we build a device that can turn a small signal into a big one...amplify it.
- Let's look back at the diode...the PN junction.
- What if stacked yet another layer of N-type to the first.
- Now we won;t get any current through...two opposing diodes.
- However, if we add a third electrode here, we can manipulate the material underneath...just with the electric field (there is an insulator between them).
- Let's see what happens if we apply a negative voltage...not much.
- However, if we apply a positive voltage, we start to attract N charge carriers...and eventually, they reach each other....creating a channel of conductivity.
- Now we can use an external source to push much more current through the device, controlled by a tiny signal, using no current at all, at the Gate.
- If we engineer to turn on quickly, we get an electronic switch.
- If we engineer to turn on slowly, ideally linearly, then we get a big signal that reflects our small signal...we get an amplifier.
- These devices are called MOSFETs...and we have figured out how to make many of them.
- They come in two flavours. N-channel (NPN) and P-channel (PNP). N-channel turn on with a positive voltage and P-channel with a negative voltage.
- There symbol is this.

## Connections

## Lesson

- **Task**: Measure the threshold voltage that opens your MOSFET gate. Compare it to the "expected" range listed in the 
    - The datasheet for your MOSFET can be found here: [IRF510](../../../boxes/transistors/_resources/datasheets/IRF510.pdf)
> The threshold for when current starts to flow through your MOSFET ("Gate-Source Threshold Voltage") should be between 2 to 4 Volts. However, the amount of current it allows will rise rapidly up to (and beyond) 10 Volts. Check the datasheet (Figure 3). 
