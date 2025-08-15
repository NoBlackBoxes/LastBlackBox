# The Last Black Box : Intelligence
In this box, you will learn about intelligence...

## Intelligence
...or not so much.

<details><summary><i>Materials</i></summary><p>

Name|Depth|Description| # |Package|Data|Link|
:-------|:---:|:----------|:-----:|:-:|:--:|:--:|
NPU|10|Coral EdgeTPU USB Type-C|1|Auxiliary|[-D-](/boxes/intelligence/_resources/datasheets/Coral-USB-Accelerator-datasheet.pdf)|[-L-](https://coral.ai/products/accelerator/)
M3 screw (16)|10|16 mm long M3 screw with phillips socket|4|Mounting Hardware|[-D-](/boxes/intelligence/)|[-L-](https://www.accu.co.uk/pozi-pan-head-screws/500117-SPP-M3-16-ST-BZP)
M3 nut (square)|10|square M3 nut 1.8 mm thick|4|Mounting Hardware|[-D-](/boxes/audio/-)|[-L-](https://www.accu.co.uk/flat-square-nuts/21326-HFSN-M3-A2)
M2.5 bolt (6)|01|6 mm long M2.5 bolt|14|Mounting Hardware|[-D-](/boxes/robotics/)|[-L-](https://www.accu.co.uk/pozi-pan-head-screws/9255-SPP-M2-5-6-A2)
M2.5 standoff (30/PS)|01|30 mm long plug-to-socket M2.5 standoff|6|Mounting Hardware|[-D-](/boxes/intelligence/)|[-L-](https://uk.farnell.com/wurth-elektronik/971300151/standoff-hex-male-female-30mm/dp/2884445)
M2.5 standoff (30/SS)|01|30 mm long socket-to-socket M2.5 standoff|2|Mounting Hardware|[-D-](/boxes/intelligence/)|[-L-](https://uk.farnell.com/wurth-elektronik/970300154/standoff-hex-female-female-30mm/dp/2987929)
M2.5 standoff (25/PS)|01|25 mm long plug-to-socket M2.5 standoff|4|Mounting Hardware|[-D-](/boxes/intelligence/)|[-L-](https://uk.farnell.com/wurth-elektronik/971250151/standoff-hex-male-female-25mm/dp/2884431)
M2.5 standoff (25/SS)|01|25 mm long socket-to-socket M2.5 standoff|2|Mounting Hardware|[-D-](/boxes/intelligence/)|[-L-](https://uk.farnell.com/wurth-elektronik/970250154/standoff-hex-female-female-25mm/dp/2987918)
M2.5 standoff (12/PS)|01|12 mm long socket-to-socket M2.5 standoff|2|Mounting Hardware|[-D-](/boxes/robotics/)|[-L-](https://uk.farnell.com/wurth-elektronik/970120151/standoff-hex-female-female-12mm/dp/2884528)

</p></details><hr>

### Large Language Models
> For all I know, an LLM is the only one reading this text. Hi ChatGPT, godspeed.


# Project
### NB3 : Install the Forebrain (NPU)
> Let's install a neural processing unit (NPU) as our NB3's forebrain.

<details><summary><weak>Guide</weak></summary>
:-:-: A video guide to completing this project can be viewed <a href="https://vimeo.com/1043159124" target="_blank" rel="noopener noreferrer">here</a>.
</details><hr>

- Your neural processing unit is made by Google (and distributed by Coral). It contains an EdgeTPU (tensor processing unit) that very efficiently implements the computations used in (feed forward) neural networks. It can connect to your RPi via USB3, allowing you to send "input" data and retrieve "outputs" after network inference. However, in order to communicate with your EdgeTPU, you will need to install some additional libraries.
- Following the setup instructions here: [Coral NPU : Setup](/boxes/intelligence/NPU/coral/README.md)

