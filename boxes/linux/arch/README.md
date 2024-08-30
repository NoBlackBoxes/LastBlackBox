# Linux : Arch

Let's install Arch Linux.

## Installation

These instructions describe setting up a new Arch Linux installation on your "Host" computer (laptop or desktop). The latest official installation guide can be found [here](https://wiki.archlinux.org/title/Installation_guide).

1. Download latest ISO: [Arch Downloads](https://archlinux.org/download/)
2. Write to USB flash drive: Use [Etcher](https://www.balena.io/etcher/)
3. Boot the USB drive: Boots to root user command line
4. Change keyboard layout if necessary (default: US)

    ```bash
    # List all keymaps
    localectl list-keymaps

    # Load correct keymap
    loadkeys <your keymap>
    # loadkeys uk
    ```

5. Connect to the internet

    ```bash
    # For WiFi
    iwctl --passphrase passphrase station device connect SSID
    # iwctl --passphrase <password> station wlan0 connect <WiFi name>

    # Wired connections should just work (via DHCP)
    ```

6. Update clock using network time protocol

    ```bash
    timedatectl set-ntp true
    ```

7. Partition disks: ***This will delete the entire drive***

    ```bash
    # List all disks
    fdisk -l

    # Select your drive
    fdisk <your drive> # e.g. /dev/sda
    ```

    - *(From within fdisk command prompt)*
    - Delete all exisiting partitions with command "d"
    - Create ESP parition (for UEFI systems)
      - Create new partion with command "n"
      - Partition number: 1 (default)
      - First sector: 2048 (default)
      - Last sector: +512M
      - (Remove signature, if requested)
    - Change partition type to 'EFI System'
      - Change type with command "t"
      - (Selects partition 1)
      - Partition type or alias: 1
    - Create swap partition
      - Create new partion with command "n"
      - Partition number: 2 (default)
      - First sector: ?? (choose default)
      - Last sector: +4096M
      - (Remove signature, if requested)
    - Create root partition
      - Create new partion with command "n"
      - Partition number: 3 (default)
      - First sector: ?? (choose default)
      - Last sector: ?? (choose default)
      - (Remove signature, if requested)
    - Write partitioning commands with command "w"

8. Create file systems on the new partitions
  
    ```bash
    # Create FAT32 file system on EFI partition
    mkfs.fat -F32 /dev/[your drive - parttion 1]p1

    # Create Swap file system on SWAP partition
    mkswap /dev/[your drive - parttion 2]p2

    # Create EXT4 file system on ROOT partition
    mkfs.ext4 /dev/[your drive - parttion 3]p3
    ```

9. Mount the file systems

    ```bash
    mount /dev/<your drive - parttion 3> /mnt
    mkdir /mnt/boot
    mount /dev/<your drive - parttion 1> /mnt/boot
    swapon /dev/<your drive - parttion 2>
    ```

10. Select download mirror site

    ```bash
    pacman -Syy # (sync)
    pacman -S reflector # Install reflector

    # Rate mirrors and update list
    reflector --verbose --latest 25 --sort rate --save /etc/pacman.d/mirrorlist
    ```

11. Install essential packages

    ```bash
    pacstrap -K /mnt base linux linux-firmware vim sudo <intel-ucode/amd-ucode>
    ```

12. Generate FSTAB

    ```bash
    genfstab -U /mnt >> /mnt/etc/fstab
    ```

13. Configure system in Chroot

    ```bash
    # Change root
    arch-chroot /mnt
    
    # Set timezone
    ln -sf /usr/share/zoneinfo/Europe/London /etc/localtime

    # Run hwclock(8) to generate /etc/adjtime:
    hwclock --systohc

    # Generate locales
    locale-gen

    # Create locale.conf
    vim /etc/locale.conf
    # - add LANG=en_GB.UTF-8

    # Create vconsole.conf
    vim /etc/vconsole.conf
    # - add KEYMAP=gb    

    # Create the hostname file
    vim /etc/hostname
    # - add <your hostname>

    # Edit host file
    vim /etc/hosts
    # - add 
    #   127.0.0.1	localhost
    #   ::1		    localhost
    #   127.0.1.1	<your hostname>

    # Initramfs
    mkinitcpio -P

    # Set up root passwd
    passwd

    # Install EFI bootloader
    pacman -S grub efibootmgr

    # Mount the ESP partition you had created
    mount /dev/[your drive]p1 /boot

    # Install grub and configure
    grub-install --target=x86_64-efi --bootloader-id=GRUB --efi-directory=/boot
    grub-mkconfig -o /boot/grub/grub.cfg

    # Create user
    useradd -m kampff
    passwd kampff

    # Add user to /etc/sudoers
    vim /etc/sudoers
    # - add <USER> ALL=(ALL) ALL

    # Install Xorg
    pacman -S xorg

    # Install KDE (plasma)
    pacman -S plasma
    
    # Install KDE (applications) ???? - Should be more minimal (Netowrk, firefox)
    pacman -S kde-system-meta
    pacman -S kde-utilities-meta
    pacman -S kde-graphics-meta
    pacman -S kde-multimedia-meta

    # Install development tools
    pacman -S base-devel
    pacman -S valgrind
    pacman -S gcc-fortran

    # Install required packages for VK
    pacman -S ffmpeg
    pacman -S minicom
    pacman -S libusb
    pacman -S ncurses
    pacman -S rsync
    pacman -S cpio
    pacman -S wget
    pacman -S bc
    pacman -S dtc
    pacman -S mtools

    # Install chromium
    pacman -S chromium

    # Install utilities
    pacman -S git
    pacman -S usbutils
    # pacman -S code - need Mirosoft extensions...use their binary version
    
    # Enable display and network manager
    systemctl enable sddm.service
    systemctl enable NetworkManager.service
    ```

14. Shutdown, remove USB, hope for the best!