# Install VS Code on your Chromebook
Open a (Linux) terminal and run the following command
```bash
dpkg --print-architecture
```
It will print the name of your Chromebook's CPU architecture

If your CPU architecture is **x86** or **amd64** or **x64**, then run the following commands:
```bash
# Download VS Code (x64 version)
wget "https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-x64" -O "code-download.deb"

# Install Gnome Keyring
sudo apt-get install gnome-keyring

# Install VS code
sudo apt install ./code-download.deb
# - Follow the on-screen instructions
```

If your CPU architecture is **arm64** or **aarch64**, then run the following commands:
```bash
# Download VS Code (arm64 version)
wget "https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-arm64" -O "code-download.deb"

# Install Gnome Keyring
sudo apt-get install gnome-keyring

# Install VS code
sudo apt install ./code-download.deb
# - Follow the on-screen instructions
```

If your CPU architecture is **arm32** or **aarch32** or **armhf**, then run the following commands:
```bash
# Download VS Code (armhf version)
wget "https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-armhf" -O "code-download.deb"

# Install Gnome Keyring
sudo apt-get install gnome-keyring

# Install VS code
sudo apt install ./code-download.deb
# - Follow the on-screen instructions
```
