# The Last Black Box : Power
In this box, you will learn about power...

## Power
Running more capable software requires a faster computer, which requires more electrical power. We will now explore how power supplies work and then install one on your NB3.

<details><summary><i>Materials</i></summary><p>

Name|Depth|Description| # |Data|Link|
:-------|:---:|:----------|:-:|:--:|:--:|
NB3 Power Board|01|Regulated DC-DC power supply (5 Volts - 4 Amps)|1|[-D-](/boxes/power/NB3_power)|[-L-](VK)
Power Cable|01|Custom 4 pin NB3 power connector cable|1|[-D-](/boxes/power/)|[-L-](VK)
M2.5 standoff (7/PS)|01|7 mm long plug-to-socket M2.5 standoff|4|[-D-](/boxes/power/)|[-L-](https://uk.farnell.com/wurth-elektronik/971070151/standoff-hex-male-female-7mm-m2/dp/2884371)
M2.5 bolt (6)|01|6 mm long M2.5 bolt|4|[-D-](/boxes/robotics/)|[-L-](https://www.accu.co.uk/pozi-pan-head-screws/9255-SPP-M2-5-6-A2)
M2.5 nut|01|regular M2.5 nut|4|[-D-](/boxes/power/-)|[-L-](https://www.accu.co.uk/hexagon-nuts/456430-HPN-M2-5-C8-Z)
12V DC Power Supply|01|12 V AC-DC transformer (UK/EU/USA plugs)|1|[-D-](/boxes/power/)|[-L-](https://www.amazon.co.uk/gp/product/B09QG4R1R4)
Battery|01|NiMH 9.6V 8-cell 2000 mAh battery|1|[-D-](/boxes/power/)|[-L-](https://www.amazon.co.uk/dp/B091H9ZFSF)
Battery Cable|01|Barrel Jack to Tamiya Plug|1|[-D-](/boxes/power/)|[-L-](VK)
Battery Charger|01|NiMH battery charger (UK plug)|1|[-D-](/boxes/power/)|[-L-](https://www.amazon.co.uk/dp/B089VRXKWY)
Velcro Patch|01|Velcro adhesive|1|[-D-](/boxes/power/)|[-L-](https://www.amazon.co.uk/50mmx100mm-Adhesive-Strips%EF%BC%8CExtra-Mounting-Organizing/dp/B0CKVNM69R)

</p></details><hr>

#### Watch this video: [DC-DC Converters](https://vimeo.com/1035304311)
> How does efficient DC to DC conversion work? Buck and Boost.


# Project
### NB3 : Power Supply
> Let's install a DC-DC power supply on our NB3.

<details><summary><weak>Guide</weak></summary>
:-:-: A video guide to completing this project can be viewed <a href="https://vimeo.com/1035306761" target="_blank" rel="noopener noreferrer">here</a>.
</details><hr>


**TASK**: Add a (regulated) 5 volt power supply to your robot, which you can use while debugging to save your AA batteries and to provide enough power for the Raspberry Pi computer.
- - *NOTE*: Your NB3_power board cable *might* have inverted colors (black to +5V, red to 0V) relative to that shown in the assembly video. This doesn't matter, as the plugs will only work in one orientation and the correct voltage is conveyed to the correct position on the body.
<details><summary><strong>Target</strong></summary>
    Your NB3 should now look like this: ![NB3 power wiring:400](../../../boxes/power/_resources/images/NB3_power_wiring.png)"
</details><hr>


### NB3 : Battery Pack
> Let's install a rechargeable NiMH battery pack on our NB3.

<details><summary><weak>Guide</weak></summary>
:-:-: A video guide to completing this project can be viewed <a href="https://vimeo.com/1042781111" target="_blank" rel="noopener noreferrer">here</a>.
</details><hr>


**TASK**: Add a NiMH rechargeable 9.6 volt battery pack to your NB3.
<details><summary><strong>Target</strong></summary>
    You should now be able to power and control your NB3 autonomously (without any cables).
</details><hr>


