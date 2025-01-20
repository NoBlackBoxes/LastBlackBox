# The Last Black Box : Intelligence
In this box, you will learn about intelligence...

## Intelligence
...or not so much.

<details><summary><i>Materials</i></summary><p>

Name|Depth|Description| # |Data|Link|
:-------|:---:|:----------|:-:|:--:|:--:|
NPU|10|Coral EdgeTPU USB Type-C|1|[-D-](/boxes/intelligence/_resources/datasheets/Coral-USB-Accelerator-datasheet.pdf)|[-L-](https://coral.ai/products/accelerator/)
M3 screw (16)|10|16 mm long M3 screw with phillips socket|4|[-D-](/boxes/intelligence/)|[-L-]()
M3 nut (square)|10|square M3 nut 1.8 mm thick|4|[-D-](/boxes/intelligence/)|[-L-](https://www.accu.co.uk/flat-square-nuts/21326-HFSN-M3-A2)

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

