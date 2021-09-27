# Connecting to your RPi over WiFi

- Download Raspberry Pi OS (latest 32-bit Lite version)
- "Burn" image to a micro-SD card

## Easy way
- Connect a monitor to the micro-HDMI port
- Connect a Keyboard and Mouse to the USB port
- Power on the Pi
- Use raspi-config to setup WiFi
- Change "hostname" to something unique
- (Optional) Use dhcpcd to set a "static" ip
- SSH (powershelll or PuTTY) to RPi via ssh pi@hostname or pi@IPAddress

## Head-less way
- Mount your micro-SD card (put it in a reader and connect to your host computer)

### Windows
 - Only the "boot" partition will be visible. ***DO NOT FORMAT the other partition.*** The "rootfs" partition is in a format (ext4) that only works with Linux operating systems.
 - Place a file named "ssh" (*note* ***no*** suffix, e.g. ssh.txt. Please make sure this is deleted as Windows sometimes adds ".txt" by default)
 - Create a file called "wpa_supplicant.conf" in the boot partition with the following contents:

 ```txt

 ```

- Power on the Pi
- Pray
- 
- SSH to RPi via ssh pi@raspberrypi or pi@IPAddress
  - What is my IP address? Well...

### MacOS
- Similar to Windows

### Linux
- Similar to Windows, but can also directly edit /etc/dhcpcd.conf in the "rootfs" partition
