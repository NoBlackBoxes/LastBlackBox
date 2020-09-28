# Setting up your Pi

Unlike the Pi you've been using for your computer so far, the SD card for this pi doesn't already have an OS on it - first things are to set this up...

1. Download the disk image from [here](https://www.dropbox.com/s/p4zqx56ep31gppf/lbbos.img?dl=0 "disk.img")
2. Follow the instructions [here](https://www.raspberrypi.org/documentation/installation/installing-images/ "instructions") to install the image on to your SD card.
3. You're done


## Connecting your pi to a new network

The distribution you installed already had the bootcamp "LastBlackBox" Wi-Fi network, but now you need to connect to a new network.  To do this, in the terminal enter the command

```bash
nano /etc/wpa_supplicant/wpa_supplicant.conf
```

which will let you edit the configuration file for WiFi-Protected Access 2 (IEEE 802.11i) protocols, though it also includes support for older protocols.  In computer science, **supplicant** refers to the client machine that wants to gain access to the network. Add a new network using the template for the LastBlackBox network. If the network you're trying to join is hidden, add `ssid_scan=1` to the declaration.  Save and close the file, at which point the Pi *should* automatically connect to the network.  If it doesn't just reboot the Pi.  If it still doesn't connect, check the security protocols for your network - you might need to specify extra settings in the `.conf` file.



## SSH into the pi

Once you've logged in and changed the password, we need to set up [ssh](https://www.ssh.com/ssh/protocol/ "ssh").  Set up the ssh server like [this](https://www.raspberrypi.org/documentation/remote-access/ssh/ "ssh-server") and then set a static IP address using this [guide](https://howchoo.com/pi/configure-static-ip-address-raspberry-pi#find-your-router-ip-address "static-ip").


***
### No static IP addresses

Not all networks will allow you to set up a static IP address.  This is because assigning static IP addresses can assign a logistical problem.  Consider that not all things that connect to the network are permanently connected, or are even connected for very long, or very often.  Every time a machine does connect, though, it requires a unique IP address (on that network) so that information can be routed to the correct place.  In the case where addresses are assigned randomly upon joining the network, the router need only compare addresses across all the machines **currently** connected.  However, if a static IP address was assigned to every computer, it would instead have to compare the current proposed address to addresses for every machine that has **ever** connected to the network.  This can be quite difficult.  If your network doesn't allow you to assign a static IP address, one option is to set up your Robot Pi as a **Wireless Access Point**, allowing your Computer Pi to communicate with it.  Good instructions are given [here](https://www.raspberrypi.org/documentation/configuration/wireless/access-point-routed.md "WAP").  I recommend running `sudo apt update` before you begin.  You can skip the section on `Enable routing and IP masquerading`, and I recommend enabling 5GHz transmission with a channel between 36 and 165.  If you already have a 5GHz network in your locale, choose a different channel to that one!

***

A static IP is address is necessary so that you can always use the same IP address to SSH into your pi.  Its now worth grabbing that IP from the `/etc/dhcpcd.conf` file you have open or by running the `ifconfig` command.

To avoid the hassle and security implications of passwords stored in plain text, I recommend you set up passwordless ssh instead.

1. Create a public key on your pi `ssh-keygen -t rsa`
2. Copy the public key across to your robot pi

```bash
ssh-copy-id -i ~/.ssh/id_rsa.pub student@<ip_address>
```

It might be helpful to copy your ip_address across to a bash variable and then pass this in

```bash
ip=<ip_address>
ssh-copy-id -i ~/.ssh/id_rsa.pub student@$ip
```

3. Make a file called `<something>.sh` containing the following code

```bash
#!/bin/bash
ssh student@<ip_address>
```

4. Make that file executable with `chmod +x <something>.sh`.  You can have a look at the permissions associated with that file using `ls -l <something>.sh`.
5. Now, connect to your Pi with `./<something>.sh`

## Communicating with your pi

The dream is to be able to communicate with your pi over ssh.  We now have an ssh tunnel set up, and can execute commands locally on the Pi.  However, it would also be nice to run a program on your local machine and simply send instructions to your robot.  This allows us to make a remote-controlled pi.  See the separate tutorial for how to set up your own server for TCP (or UDP) communication.


## Pi to Arduino communication

You can communicate with the arduino from your pi using python, using the Pyserial module.  This module should already be installed on your machine.  You will need to know what serial port to use, which you can find with `dmesg | grep "tty"`.  In this line, [`dmesg`](https://man7.org/linux/man-pages/man1/dmesg.1.html "documentation") displays all messages from the kernel ring buffer.  The kernel stores its logs in a buffer.  This buffer maintains a constant size by dropping old stuff as new stuff comes in.  This is known as a ring buffer.  `dmesg` prints the contents of this ring buffer.  We direct the output of this command to `grep` using the `|` operator (called a "pipe" in linux language) and search for the `tty` regular expression.  This will tell us about recent kernel calls to `tty` ports, and in particular we are interested in `Serial Device <> now attached to tty<>`.  This kind of operation (piping output from simple commands and then using other commands, particularly `grep` to parse) will be immensely helpful in your linux journey.


