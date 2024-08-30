# systems : lbbos : rpios

Instructions for modifying the official 32-bit Raspberry Pi OS for use with the Last Black Box.

## Setup

1. Download the most recent 32-bit desktop version (without recommended software) from the [RPi](https://www.raspberrypi.org/downloads/raspberry-pi-os/)
2. Flash the image to a micro-SD card using Etcher
3. On first boot, follow the configuration, connect to the internet, and install updates
4. Run raspi-config, enable SSH, Camera, GPU memory = 256 MB, hostname = LBB
5. Create new user: student

    ```bash
    sudo adduser student
    ```

6. Add "student" to sudo group

    ```bash
    sudo usermod -aG sudo student
    ```

7. Change autologin to "student"

    ```bash
    sudo nano /etc/lightdm/lightdm.conf
    ```

    Change the "autologin-user=pi" to "autologin-user=student"  

8. Use "raspi-config", switch to console boot, no auto-login...and reboot

9. Login as "student" and delete "pi" user

    ```bash
    sudo userdel -r pi
    ```

    - Add student mto video group

    ```bash
    sudo usermod -a -G video student
    ```    

10. Clone LBB repo

    ```bash
    mkdir -p ~/LastBlackBox
    cd ~/LastBlackBox
    git clone https://github.com/kampff/LastBlackBox repo
    ```

11. Install VSCode (actually VSCodium, the same open source version without Microsoft licenses and telemetry)

    - Download the most recent release for "armhf" (as a Debian package)

      ```bash
      cd ~/Downloads
      wget https://github.com/VSCodium/vscodium/releases/download/1.48.2/codium_1.48.2-1598439436_armhf.deb
      ```

    - Install the *.deb package using apt

      ```bash
      cd ~/Downloads
      sudo apt install ./codium_1.48.2-1598439436_armhf.deb
      ```

12. Check the VSCodium works...

13. Install useful packages

    ```bash
    sudo apt-get install vim
    ```

14. Configure i2s (sound)

    - https://learn.adafruit.com/adafruit-max98357-i2s-class-d-mono-amp/raspberry-pi-usage
