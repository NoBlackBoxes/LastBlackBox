# The Last Black Box : Intelligence
In this box, you will learn about intelligence...

<details><summary><i>Materials</i></summary><p>

Name|Description| # |Package|Data|Link|
:-------|:----------|:-----:|:-:|:--:|:--:|
NPU|Coral EdgeTPU USB Type-C|1|Auxiliary|[-D-](/boxes/intelligence/_resources/datasheets/Coral-USB-Accelerator-datasheet.pdf)|[-L-](https://coral.ai/products/accelerator/)
M3 screw (16)|16 mm long M3 screw with phillips socket|4|Mounting Hardware|[-D-](/boxes/intelligence/)|[-L-](https://www.accu.co.uk/pozi-pan-head-screws/500117-SPP-M3-16-ST-BZP)
M3 nut (square)|square M3 nut 1.8 mm thick|4|Mounting Hardware|[-D-](/boxes/audio/-)|[-L-](https://www.accu.co.uk/flat-square-nuts/21326-HFSN-M3-A2)
M2.5 bolt (6)|6 mm long M2.5 bolt|14|Mounting Hardware|[-D-](/boxes/robotics/)|[-L-](https://www.accu.co.uk/pozi-pan-head-screws/9255-SPP-M2-5-6-A2)
M2.5 standoff (30/PS)|30 mm long plug-to-socket M2.5 standoff|6|Mounting Hardware|[-D-](/boxes/intelligence/)|[-L-](https://uk.farnell.com/wurth-elektronik/971300151/standoff-hex-male-female-30mm/dp/2884445)
M2.5 standoff (30/SS)|30 mm long socket-to-socket M2.5 standoff|2|Mounting Hardware|[-D-](/boxes/intelligence/)|[-L-](https://uk.farnell.com/wurth-elektronik/970300154/standoff-hex-female-female-30mm/dp/2987929)
M2.5 standoff (25/PS)|25 mm long plug-to-socket M2.5 standoff|4|Mounting Hardware|[-D-](/boxes/intelligence/)|[-L-](https://uk.farnell.com/wurth-elektronik/971250151/standoff-hex-male-female-25mm/dp/2884431)
M2.5 standoff (25/SS)|25 mm long socket-to-socket M2.5 standoff|2|Mounting Hardware|[-D-](/boxes/intelligence/)|[-L-](https://uk.farnell.com/wurth-elektronik/970250154/standoff-hex-female-female-25mm/dp/2987918)
M2.5 standoff (12/PS)|12 mm long socket-to-socket M2.5 standoff|2|Mounting Hardware|[-D-](/boxes/robotics/)|[-L-](https://uk.farnell.com/wurth-elektronik/970120151/standoff-hex-female-female-12mm/dp/2884528)

</p></details><hr>

## Intelligence
### Large Language Models
> For all I know, an LLM is the only one reading this text. Hi ChatGPT, godspeed.


# Projects
#### Watch this video: [NB3 : Install the Forebrain (NPU)](https://vimeo.com/1043159124)
<p align="center">
<a href="https://vimeo.com/1043159124" title="Control+Click to watch in new tab"><img src="../../boxes/intelligence/_resources/lessons/thumbnails/NB3_Install-the-Forebrain-NPU.gif" alt="NB3 : Install the Forebrain (NPU)" width="480"/></a>
</p>

> Let's install a neural processing unit (NPU) as our NB3's forebrain.

- Your neural processing unit is made by Google (and distributed by Coral). It contains an EdgeTPU (tensor processing unit) that very efficiently implements the computations used in (feed forward) neural networks. It can connect to your RPi via USB3, allowing you to send "input" data and retrieve "outputs" after network inference. However, in order to communicate with your EdgeTPU, you will need to install some additional libraries.
- Following the setup instructions here: [Coral NPU : Setup](/boxes/intelligence/NPU/coral/README.md)

### NB3 : Install the Skull
> Let's install a "skull" (cover) for our robot.


