# Electrons : NB3 : Body
We will now start measuring and manipulating electricity, and first we will assemble a "prototyping platform" that also happens to be the **body** of your robot (NB3).

## [Video](https://vimeo.com/1030776673)

## Concepts
- Your NB3 body will initially serve as an electronics prototyping platform. Where we will build simple circuits and learn about electricity.
- It is shaped like a brain, actually a "simple vertebrate brain", with three main regions: forebrain, midbrain, and hindbrain. We will build systems that are analogous to the systems found in these regions of the brain as we construct are more and more sophisticated NB3.
- The NB3 is actually a large "printed circuit board" or PCB.
- It was designed using a free and open-source software called KiCAD, which you can learn about in another video, and manufactured by a PCB company.
- In short, it consists of two thin layers of copper on the top and bottom surface of a fiberglass board (which provides the stiffness). Finally, all the metal, except where I need connections to the outside world, is covered in layer of insulator called "solder mask". This material gives the board its color. You may be more familiar with the traditional "green" PCBs.
- I have designed a series of patterns on the top and bottom copper layers to make convenient electrical connections between different places on the board. You can see this routes if you look closely.
- Sometimes the routes jump between the layers using a via. A via is a metal-plated hole drilled through the board that allows our electrical connection to change sides. This makes it much easier to route many different signals around to their destinations.
- More complex PCBs use many more layers, sometimes 10 or more, in order to route thousands of connection throughout the board (in a small area).
- These copper routes terminate at a variety of connectors, which we will use to connect different devices (sensors, motors, etc.) to our NB3.
- The routes are labeled so you know which ones are connected together at different places o the board. lA to lA, etc. These are sometimes called "nets".
- One of the most useful "nets" is the power net, labeled +5V and Ov throughout the board. However, it does not have to be +5V, depends on what you connect as a power supply. The wires carrying (and returning) current from the power supply are typically wider traces, and in fact, the entire top layer of the board is connected to +5V and the bottom to 0V.
- Our first task is to prevent the NB3 body from getting scratched while we work on it and build our first circuits.
- Sliding the body around on a rough surface can scratch the thin layer of insulator and damage the even thinner copper wires. We will therefore add some little plastic (silicone) "feet" to elevate the board off your work surface. Do that now.
- Finally, we need a place to actually build our first circuits. We don't want to have to design and manufacture a "printed circuit board" and then solder all the connections each time we test a new circuit. We have therefore given you 5 "solderless" breadboards. These are "plug-n-play" circuit prototyping tools that, when you figure out how they should be used, will be very useful while you a learning about electronics, and even later, when you want to quickly test a new design.
- We will learn all about how to build circuits with solderless breadboards in another video. For now, we just need to attach them to the NB3 body.
- They come with a convenient adhesive back...which is very difficult to "unstick"...so try to get everything aligned the first time.
- Add your breadboards now.

## Connections

## Lesson

- **TASK**: Assemble the robot body (prototyping base board).
  - *Challenge*: If you are curious how the *NB3 Body* printed circuit board (PCB) was designed, then you can find the KiCAD files here: [NB3 Body PCB](../../../boxes/electrons/NB3_body). You can also watch this short introduction to PCB design with KiCAD here: [NB3-Designing PCBs with KiCAD](https://vimeo.com/??????).
> Your NB3 should now look like [this](../../../boxes/electrons/NB3_body/NB3_body_front.png). Your breadboards will be different colors...and you should have some rubber feet on the back.
