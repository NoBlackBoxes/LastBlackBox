# Systems : ChromeOS

If you would like to use a Chromebook as your host computer, then there are a few extra steps to get things setup.

## Enable Linux on your Chromebook

These instructions come from [here](https://support.google.com/chromebook/answer/9145439?hl=en-GB).

- Goto Settings -> About Chrome OS -> Developers
  - Next to "Linux development environment", click "Setup" and follow the on-screen instructions.

A terminal window will appear. This is your Linux terminal.

## Install VS Code

  ```bash
  # Make sure everything is up-to-date
  sudo apt update
  sudo apt upgrade -y

  # Check which CPU architecture your Chromebook is using
  dpkg --print-architecture

  # Download ONE of the following, depending on your architecture

  ## ...Download VS Code (amd64)
  wget "https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-x64" -O "code-download.deb"

  # ...OR

  ## ...Download VS Code (arm64)
  wget "https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-arm64" -O "code-download.deb"

  # Install Gnome Keyring
  sudo apt-get install gnome-keyring

  # Install VS code
  sudo apt install ./code-download.deb
  # - Follow the on-screen instructions
  ```

## Install the Arduino IDE

  ```bash
  # Download ONE of the following, depending on your architecture

  # ...Download Arduino 1.8.19 (amd64)
  wget "https://downloads.arduino.cc/arduino-1.8.19-linux64.tar.xz" -O "arduino-1.8.19.tar.xz"

  # ...OR

  # ...Download Arduino 1.8.19 (arm64)
  wget "https://downloads.arduino.cc/arduino-1.8.19-linuxaarch64.tar.xz" -O "arduino-1.8.19.tar.xz"

  # Extract the archive
  tar -xf arduino-1.8.19.tar.xz

  # Run the install script
  cd arduino-1.8.19
  sudo ./install.sh
  ```
