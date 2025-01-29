## Enable Linux on your Chromebook

*These instructions come from* [here](https://support.google.com/chromebook/answer/9145439?hl=en-GB).

- Goto Settings -> About Chrome OS -> Developers
  - Next to "Linux development environment", click "Setup" and follow the on-screen instructions.

A terminal window will appear. **This is your Linux terminal!**

## Install the Arduino IDE
Run these commands inside your Linux terminal. You can simply copy and paste this entire block of text.

```bash
# Make sure everything is up-to-date
sudo apt update
sudo apt upgrade -y

# Check which CPU architecture your Chromebook is using
dpkg --print-architecture
```

If your CPU architecture is **x86** or **amd64**, then run the following commands:
```bash
# ...Download Arduino 1.8.19 (amd64 or x86)
wget "https://downloads.arduino.cc/arduino-1.8.19-linux64.tar.xz" -O "arduino-1.8.19.tar.xz"

# Extract the archive
tar -xf arduino-1.8.19.tar.xz

# Run the install script
cd arduino-1.8.19
sudo ./install.sh
```

If your CPU architecture is **arm64** or **aarch64**, then run the following commands:
```bash
# ...Download Arduino 1.8.19 (am64 or aarch64)
wget "https://downloads.arduino.cc/arduino-1.8.19-linuxaarch64.tar.xz" -O "arduino-1.8.19.tar.xz"

# Extract the archive
tar -xf arduino-1.8.19.tar.xz

# Run the install script
cd arduino-1.8.19
sudo ./install.sh
```

If your CPU architecture is **arm32** or **aarch32**, then run the following commands:
```bash
# ...Download Arduino 1.8.19 (arm32 or aarch32)
wget "https://downloads.arduino.cc/arduino-1.8.19-linuxarm.tar.xz" -O "arduino-1.8.19.tar.xz"

# Extract the archive
tar -xf arduino-1.8.19.tar.xz

# Run the install script
cd arduino-1.8.19
sudo ./install.sh
```
