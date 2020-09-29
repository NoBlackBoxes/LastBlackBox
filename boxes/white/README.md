# White

Fight the dark with the light. Your white box contains all the tools you will need (and learn to use) to open black boxes.

----

<details><summary><b>Materials</b></summary><p>

Contents|Description| # |Data|Link|
:-------|:----------|:-:|:--:|:--:|
Computer (RPi4)|Raspberry Pi 4b with 4 GB RAM|1|[-D-](_data/datasheets/rpi4b_4gb.pdf)|[-L-](https://uk.farnell.com/raspberry-pi/rpi4-modbp-4gb/raspberry-pi-4-model-b-4gb/dp/3051887)
Case|Silver aluminum heatsink case (Akasa Gem Pro)|1|[-D-](https://www.akasa.co.uk/search.php?seed=A-RA09-M2S)|[-L-](https://www.amazon.co.uk/gp/product/B089GVMK37/ref=ox_sc_act_title_1?smid=AHX2VT4JA3HIX&psc=1)
Power supply|5V/3A (15W) USB Type-C power supply|1|[-D-](_data/datasheets/rpi_power_supply_5V3A.pdf)|[-L-](https://uk.farnell.com/raspberry-pi/sc0212/rpi-power-supply-usb-c-5-1v-3a/dp/3106940)
Cable (HDMI)|micro to full HDMI cable (1 m)|1|[-D-](_data/datasheets/hdmi_cable_micro_1m.pdf)|[-L-](https://uk.farnell.com/raspberry-pi/t7689ax/cable-micro-hdmi-hdmi-plug-1m/dp/3107125)
SD card|32 GB micro SD card|1|[-D-](_data/datasheets/SanDisk-SDSQXCG-032G-GN6MA-datasheet.pdf)|[-L-](https://www.amazon.co.uk/dp/B06XYHN68L/ref=twister_B07J6Z8HHM?_encoding=UTF8&th=1&qty=15)
Multimeter|(Sealy MM18) pocket digital multimeter|1|[-D-](_data/datasheets/MM18_DFC0123042.pdf)|[-L-](https://www.ffx.co.uk/tools/product/Sealey-Mm18-5051747848412-Pocket-Multimeter)
Test Lead|Alligator clip to 0.64 mm pin (20 cm)|2|-|[-L-](https://www.amazon.co.uk/Oiyagai-Alligator-Crocodile-Arduino-Raspberry/dp/B07CXTSX8R/ref=sr_1_2?dchild=1&keywords=Oiyagai-Alligator-Crocodile-Arduino-Raspberry&qid=1598887302&s=computers&sr=1-2)
Screwdriver|Phillips driver (size #1)|1|[-D-](_data/datasheets/screwdriver_phillips_1.pdf)|[-L-](https://uk.farnell.com/wera/118024/screwdriver-precision-ph1x80mm/dp/1337805)
Screwdriver|Slotted driver (tip 3 mm)|1|[-D-](_data/datasheets/screwdriver_slotted_3mm.pdf)|[-L-](https://uk.farnell.com/wera/118010/screwdriver-precision-slot-3-0x80mm/dp/1337801)
Camera (RPiHQ)|Raspberry Pi high quality 12.3 MP camera|1|[-D-](_data/datasheets/rpi_camera_hq.pdf)|[-L-](https://uk.farnell.com/raspberry-pi/rpi-hq-camera/rpi-high-quality-camera-12-3-mp/dp/3381605)
Lens (6mm)|CS-mount wide-angle lens (F1.2/FL 6 mm)|1|[-D-](_data/datasheets/lens_cs_mount_6mm.pdf)|[-L-](https://uk.farnell.com/raspberry-pi/rpi-6mm-lens/rpi-6mm-wide-angle-lens/dp/3381607)
Tripod|Small tripod with camera mount (white)|1|[-D-](https://www.manfrotto.com/uk-en/pixi-mini-tripod-white-mtpixi-wh/)|[-L-](https://www.amazon.co.uk/Manfrotto-PIXI-Mini-Tripod-White/dp/B00GUND8XM)

Required|Description| # |Box|
:-------|:----------|:-:|:-:|
Monitor|Display with HDMI input|1|-|
Keyboard|USB keyboard|1|-|
Mouse|USB mouse|1|-|

</p></details>

----

## Goals

To open this box, you must complete the following tasks:

1. Setup the LBB host computer (install it in a "heatsink case", attach an HDMI monitor, keyboard, and mouse, and apply power)
2. Change the default password of the "student" user
3. Pull (i.e. sync) the latest version of the "LastBlackBox" repo
4. Setup your multimeter (install the batteries)

To explore this box, you should attempt the following challenges:

1. Read the CPU temperature of your RPi
2. Run a benchmarking test to "stress test" your RPi, while monitoring (or logging) the CPU temperature

----

## LBB Host Computer

You will be able to do all of the course exercises using your LBB host computer. Eventually, we will build another version of the host...so for now...this "black box" goes in the white box.

The LBB Host computer runs a modified version of "raspios" (Raspberry Pi OS). We have installed some extra software and loaded some useful data, but don't worry, we will explain exactly how we did this later in the course (when we build our own "lbbos" (LastBlackBox OS) from scratch).

The SD card included in this box should already have the OS image installed. If not, you can download the latest LBB "raspios" version as a disk image from here: [LBBOS](https://www.dropbox.com/s/p4zqx56ep31gppf/lbbos.img?dl=0)

### LBBOS

The Last Black Box operating system (lbbos) is a modified version of Debian Linux and derived from the "Raspbian" distribution. The version of LBBOS included in your white box is a direct clone of a running system. It has user accounts setup, software installed (including VSCodium), and has already cloned some useful GitHub repositories.

The default user is "student" with a default password of "lastblackbox". *CHANGE THE DEFAULT PASSWORD.*

To do this the "linuxy way" (which you will soon become fond of if you are not already), run this in your RPi's terminal:

```bash
passwd
```

This will walk your through a password update. *DO NOT FORGET YOUR PASSWORD!* But if you do, it's not the end of the world.

### What happens when I turn on my Raspberry Pi?

You supply the Pi with 5V of power and about 3A. As we will see, this is a reasonably low voltage and a reasonable high amount of current. The actual "chip" that runs code on the Pi uses 3.3V. This means a power conversion takes place between the power supply and the board itself. This conversion is lossy, or not perfect, so heat is generated when this happens. When we place the Raspberry Pi in it's case, we added some heat paste to help dissipate the heat that's generated during this power conversion.

Connect the mouse and keyboard to the black, USB2.0, slots. These are lower bandwidth (slower), which doesn't matter as much for the peripherals. We'll save the faster USB3.0 slots for heftier peripherals in later boxes.

When the Pi "boots", it runs its bootloader (already present on the chip) which checks for hardware and memory, finds the files on the SD card (hopefully) and starts the startup program to kick off the running of the machine as we know it. It first loads what's called the kernel, and then bootstraps itself from there.

If you do not have the modified OS, look below for steps to get it installed on your SD card:

<details><summary><b>Cloning LBBOS</b></summary><p>

To clone an exact copy of a (functional) LBBOS:

1. Remove the SD card from your RPi
2. Insert it into a machine running Debian-based Linux (e.g. Ubuntu)
3. Identify the name of the SD card device using *fdisk*:

    ```bash
    sudo fdisk -l
    ```

4. Unmount any partitions that were mounted upon inserting the SD card

5. Copy the SD card contents to an image file on your computer:

    ```bash
    sudo dd if=<name or your SD card device> of=lbbos_backup.img
    ```

6. The resulting image will be the size of the SD card. This is usually excessive and can be reduced before burning onto another SD card. Shrinking the image is accomplished by a bash script called [PiShrink](https://github.com/Drewsif/PiShrink).

    ```bash
    wget https://raw.githubusercontent.com/Drewsif/PiShrink/master/pishrink.sh
    chmod +x pishrink.sh
    sudo ./pishrink.sh -v -p lbbos_backup.img lbbos.img
    ```

7. Copy the new (shrunken) image to a new SD card using [Etcher](https://www.balena.io/etcher/).

</p></details>

### Pull the LBB Github repo

Now we need to get the most recent version of this repo. Open up a terminal and navigate (using `cd` or change directory) to the place you want to put the course materials. I suggest you use "~", a shortcut for your "home" folder:

```bash
cd ~/LastBlackBox
```

In this "root" folder, there's a folder called "repo". Go there:

```bash
cd repo
```

and then "pull" the latest version of the repo:

```bash
git pull
```

You might see the additions to some files printed out. You should be good to go!

<details><summary><b>What is git?</b></summary><p>

Git solves the problem of *version control*. Git is a way of "tracking changes" between version of code such that we can "revert" the last working version of your code so that you feel free to destroy and rebuild your code at will, knowing all the while that you'll have a working version. Git is a universe unto itself, so we'll leave it here for now and use Git and it's "cloud" based backup solution GitHub (this allows you to "push" your code to a server somwhere owned by Microsoft by making a Github account).

</p></details>

### Notes

- Use a different multimeter (that can measure smaller currents and has over-current protection circuit (vs. a fuse)). For now, explain better how to avoid blowing the fuse when measuring current.

- White Box version 1.0: Rpi 4b 4Gb, custom passive heat-sink enclosure (extruded aluminum from Takachi) with LBB logo. Custom RPi Hat with two analog output channels (> 100 kSamples/sec...up to 500 kS per channel) +/- 5 V, two analog input channels (> 100 kSamples/sec, upto 1 MS per channel) +/- 5V. This same hat could be used with the NB3 as well to output to two speakers and monitor two mics. They should be character devices or block? Should they output buffers? Should they interface via PCIexpress/USB/I2C? Could use I2S? Alternatively, could have more channels (4 input, 4 output)?

- Power board: NB3 power supply board, charges 4 to 6 NiMH AA cells via USB type-C (or, could use a 12V/2A wall wart barrel jack), 5V switching regulator to power RPi via PoGo pins? Outputs breadboard supply (5V, 500 mA), and two motor outputs (5V, 250 mA)

----
