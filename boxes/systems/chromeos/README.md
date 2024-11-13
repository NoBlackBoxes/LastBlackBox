# Systems : ChromeOS

If you would like to use a Chromebook as your host computer, then there a few extra steps tp get things setup.

## Enable Linux on your Chromebook

These instructions come from [here](https://support.google.com/chromebook/answer/9145439?hl=en-GB).

- Goto Settings - About Chrome OS - Developers
  - Next to "Linux development environment", click "Setup"

A terminal window will appear. This is your Linux terminal.

- Install VS Code
  - Download VS Code from [here](https://code.visualstudio.com/download).


```bash
# Make sure everything is up-to-date
sudo apt update
sudo apt upgrade -y

# Check which CPU architecture your Chromebook is using
dpkg --print-architecture

# Download one of the following, depending on your architecture

# Download VS Code (amd64)
wget "https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-x64" -O "code-download.deb"

# Download VS Code (arm64)
wget "https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-arm64" -O "code-download.deb"

# Install Gnome Keyring
sudo apt-get install gnome-keyring

# Install VS code
sudo apt install ./code-download.deb
# - Follow the onscreen instructions
```

- Install the Arduino IDE

```bash
# Download Arduino 1.8.19 (arm64)
wget "https://downloads.arduino.cc/arduino-1.8.19-linuxaarch64.tar.xz" -O "arduino-1.8.19.tar.xz"

# Extract the archive
tar -xf arduino-1.8.19.tar.xz

# Run the install script
cd arduino-1.8.19
sudo ./install.sh
```