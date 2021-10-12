# The Last Black Box Bootcamp: Day 3 - Systems and Netowrks

## Morning

----

### NB3 Build (midbrain)

- Watch this video: https://vimeo.com/627777644
- *Task 1*: Mount a Raspberry Pi on your robot
- *Task 2*: Connect to your Raspberry Pi from your Host computer (via SSH)

  - This pinout of the Raspberry Pi GPIO might be useful: [Raspberry Pi GPIO](resources/images/rpi_GPIO_pinout.png)
  - Examples of files (*ssh* and *wpa_supplicant.conf*) you need to create on the "boot" partition of the micro-SD card can be found here: [boot](resources/connecting/boot)

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

- Watch this video: (Introduction to OSes)
- *Task 3*: Python practice

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

- Ears
