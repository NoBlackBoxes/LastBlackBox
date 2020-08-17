# Sensors

Computers (and brains) work with voltages. In order for a computer to understand other signals in the environment (light, sound, pressure, heat, etc.), then these signals must first be converted into a voltage. This conversion is called transduction and the thing that does it is a transducer. However, for our purposes, we will call these transduction devices *sensors*.

----

Contents            |Description
:-------------------|:-------------------------
Thermistor          |
Photoresistor       |LDR
FSR                 |Force sensing resitor
Microphone          |Piezeo

Required            |Description
:-------------------|:-------------------------
Battery             |[*not included*] Standard AA alakine battery
Multimeter          |[White Box] Sealy MM-18 digital multimeter

----

## Measuring Heat

Heat is the average kinetic energy within a material. When some materials change their temperature, their resiatnce to electric current also changes. We can use these materials to make "thermistor", a temperature dependent resistor.

We can convert a thermistor's change in resistance to a change in voltage using a voltage divider, where one of the reistors is replaced with a thermistor. The voltage between them will now change with temperature, and we will have a temperature sensor that can talk to a computer.

![Heat Sensor](images/heat_sensor.png)

Your body senses temperature differently. Special proteins, called TRP (pronounced "trip") channels straddle the insulating membrane surrounding a neuron. The neuron is more negative on the inside than outside, thanks to the Sodium-Potassium pumps, and thus positve ions outside would love to enter the cell. The TRP channels are usually closed, but will change their structure and open a small hole (i.e. channel) when they reach a specific temperature. When they open, then ions can flow into the cell, changing its voltage. This is how the nervous system converts temperature to voltage. 

![TRP Channels](images/trp_channels.png)

There a different types TRP channels sensitive to different temperatures, some for hot and very hot, and others for cold and very cold. There is an interesting consequence of relying specific channels for detecting specific temperatures. The TRP channels are proteins with part of their structure sticking out of the cell and exposed to whatever happens to be around. Some chemicals are able to bind to this exposed bit of the protein and change its structure, causing it to open. If this happens, then your brain will have no way of distinguishing this kinds of opening from when it encounters a change in temperature. A famous chemical that binds to the TRP channel that reports "very hot" temperatures is called capsacin, and it is found in spicy foods. (Yes, the "heat" of spicy foods are an illusion of your bodie's temperature sensing system). Another famous chemical is menthol, which binds to "cold" sensing TRP channels. Another, slightly less well known, is mustard oil, which binds to the "very cold" TRP channels. Mustard oil is found foods like wasabi and horseradish, and this "extreme cold illusion" explains the particular, and intense, sensory response.

### Exercise: Build a heat sensor

### Exercise: Convince your brain

- Get a very hot chilli pepper. Eat it. Tell your brain it is just an illusion. If your brain doesn't believe you.

----

## Measuring Light

Light

### Exercise: Build a light sensor

----

## Measuring Touch

Pressure

### Exercise: Build a pressure sensor

----

## Measuring Sound

Sound

### Exercise: Build a sound sensor (i.e. microphone)
