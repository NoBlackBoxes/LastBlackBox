# The Last Black Box Bootcamp: Day 3 - Systems and Netowrks

## Morning

----

### NB3 Build (midbrain)

- Watch this video: [NB3 hindbrain](https://vimeo.com/627777644)
- *Task 1*: Mount a Raspberry Pi on your robot (and connect its power inputs, 2x5V and 2x0V, to the correct GPIO pins...please *double-check* the pin numbers)
  - This pinout of the Raspberry Pi GPIO might be useful: [Raspberry Pi GPIO](resources/images/rpi_GPIO_pinout.png)
- *Task 2*: Copy a 32-bit version of the Raspberry Pi OS (operating system) to your micro-SD card.
  - You can download the most recent version here: [RPi OS Download](https://www.raspberrypi.com/software/operating-systems/)
    - The "Lite" version is sufficent, but the "Desktop" version will include some software that could be useful in later steps (i.e. debugging the WiFi connection).
  - Use a program (such as [Etcher](https://www.balena.io/etcher/) to copy the downloaded image to your micro-SD card.
  - ***Before*** you insert the micro-SD card into the Raspberry Pi, complete the first steps of the next Task.
- *Task 3*: Connect to your Raspberry Pi from your Host computer (via SSH)
  - This will require adding two files to the "boot" partition on the micro-SD card that tell your NB3's Raspberry Pi how to connect to a WiFi network.
    - Examples of the files (*ssh* and *wpa_supplicant.conf*) that you need to add can be found here: [boot](resources/connecting/boot)
      - *Note*: you must change the Wifi name (SSID) and password in the *wpa_supplicant.conf* file to match your local WiFi network.
      - *Note*: Make sure the "ssh" file has no extension (e.g. *.txt*), which sometimes is added by default.
  - What is SSH? It stands for "**s**ecure **sh**ell". It is a program that runs on your "host" computer that gives you access via a network to the terminal (shell) of a remote computer (e.g. your NB3).
    - *Windows*: you can use PuTTY and connect via the "ssh" protocol
    - *MacOS/Linux*: you can use *ssh* from the command line of a terminal
  - ***You will need to find out the IP address of your Raspberry Pi*** before you can connect to it via SSH. How do you find this out?
    - *Make sure your host computer is connected to the ***same*** network as your RPi.*
    - If you have access to your WiFi router, then you check for any *new* devices that connect when you turn on your RPi...that will tell you the IP address that your RPi was "assigned" when it connected.
    - If you have a micro-HDMI cable and a spare monitor/TV, then you can connect it before the RPi boots and watch the "scrolling text logs" of the Linux OS while it boots up. At the very end, there will be a line that says..."connected, etc. IP addres: XXX.XXX.XXX.XX". Then you know the IP address.

- *Task 4: Update your Operating System
  - When you are able to connect to your RPi, then run the following commands from the terminal prompt. (*These will update and upgrade the RPi OS and libraries to the latest version, which may help prevent version conflicts in the subsequent exercises.*)

```bash
# Change the default password for the "pi" user
passwd

# Update the "package manager" (so it knows where to looks for any upgrades)
sudo apt update

# Upgrade the OS (and installed libraries) using the "package manager"
sudo apt upgrade

# Install git
sudo apt-get install git

# Clone the LastBlackBox git repository
git clone https://github.com/NoBlackBoxes/LastBlackBox
```

### Operating Systems

- Watch this video: [Operating Systems](https://vimeo.com/630456267)
- *Task 5*: Python practice

----

## Afternoon

----

### Networks

- Live Lecture: "Communication, protocols, and the internet"
- *Task 1*: Design and implement a protocol for communicating between your midbrain and hindbrain
- ***Project***: Build a remote control robot

----

## *Evening*

----

### Ears

Let's let your NB3 hear. Let's add some ears (digital MEMS microphones).

- Watch this video: [NB3 Ears](https://vimeo.com/630461945)
- Use Python to record sound from each microphone (left and right).
  - Can you localize the sound source?...tricky.
