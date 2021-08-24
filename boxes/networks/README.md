# Networks

Things started getting really interesting when organisms began to interact with each other.

Transfer of information is really at the cornerstone of everything we do: the first step preceding any fancy computation, is data aquistion Furthermore, as we live in a noisy world, communication needs to be robust.

In this box, we being to open up the world of the internet and start communicating between our NB3's. We then move onto communicating with the outside world. 

----

<details><summary><b>Materials</b></summary><p>

Contents|Description| # |Data|Link|
:-------|:----------|:-:|:--:|:--:|
Cable (Ethernet)|RJ45 cact5e ethernet patch cable (1 m)|1|[-D-](_data/datasheets/ethernet_cable_1m.pdf)|[-L-](https://uk.farnell.com/pro-signal/ps11074/lead-patch-cat-5e-1-00m-black/dp/1734943)

</p></details>

----
## Goals

By the end of this piece, you should have a grasp of:

### Concepts
* How the internet generally works (broad goal)
* Internet networks and their hierachy
* IP addresses
* How IP addresses assigned and linked to urls

On a practical side, we should feel comfortable:

### Tasks and challenges

* communicating between our NB3 from our computer/desktop
* communicating between our bot and brain NB3 from within a local network
* setting up a local network of our own that we control
* communicating to the outside world with our NB3/ setting up our own server

----
## IP addresses

Once you connect to a network (e.g.: ethernet network), you get a unique name. This is called an IP address. (IP = internet protocol).

If you're connected to different networks, you'll have multiple IP addresses.

All internet addresses are of the form: nnn.nnn.nnn.nnn**** where nnn must be a number from 0 to 255. 


### Exercise: Find your IP addresses

1. Determine the IP address of your NB3 RPi

```bash
ifconfig
```
This shows all the active interfaces you have. The abbrevaition `eth0` stands for ethernet and `wlan0` for WiFi networks. The IP address is what follows `inet`. 
Another useful shorthand is:
```bash
hostname -I
```

2. Determine the IP address of your router (needed for later)

```bash
route -ne
```
The local IP address of your network router (the thing that brings the internet to us) is also known as the gateway IP.

The `Iface` column lists the names for each connection. The IP addresses under the `Gateway` column is what were after. This is what we're after.

3. Determine the IP addresses of your domain name servers (also useful later)

```bash
cat /etc/resolv.conf
```

Your Pi sends the domain names you enter into your browser (i.e. www.google.com) to domain name servers, which convert the domain names to IP addresses (i.e. 8.8.8.8). Your Pi then uses the IP address to access the website’s server.


#### References
- https://www.circuitbasics.com/how-to-set-up-a-static-ip-on-the-raspberry-pi/


----
## SSH

Computers within a same network (e.g.: same ethernet) can easily communicate with each other.

Connect your NB3 and RPi to a same network and communicate via ssh. `ssh` stands for secure shell.


### Exercise: Setup up SSH connection between your computer and NB3

1. make sure ssh is activated on your RPI (using `raspi-config`)
2. from your computer: 
```bash
ssh student@<IP-ADDRESS-OF-NB3>
```
3. this should give you access to the terminal of your RPI (no more clucky desktops needed!)

----
## Wireless Networks

Intro to follow

### Exercise: Connect to a Wireless Network

1. A first useful thing to do first is to check what wireless networks your NBE is detecting:
```bash
 sudo iwlist wlan0 scan
```

2. To connect to WiFi you should only need change the `wpa_supplicant.conf` file by adding a few lines to the existing test (don't delete what's already there!). WPA stands for WiFi Protected Access - supplicant refers to the client machine that wants to gain access to the network).
The main command will be:
```bash
sudo nano /etc/wpa_supplicant/wpa_supplicant.conf
```
3. check out for nice [instructions](https://www.raspberrypi.org/documentation/configuration/wireless/wireless-cli.md)
    
    - we'll need to select what interface we're setting up a static ip address for: 

```bash
interface wanl0
static ip_address: you-get-to-pick-this (as long as no one else has this IP)
```



#### References
- https://www.raspberrypi.org/documentation/configuration/wireless/wireless-cli.md

----

## Dynamic Host Configuration Protocol (DHCP) and setting up a static IP address  

We mentioned robustness at the beginning. It turns out that as IP addresses are limited, service providers (more or them later), reallocate IP addresses every so often. However, this can be very frustrating as then you never know by what name to call your colleague. To overcome this, certain networks allow you to have a static IP address (which is well, static...).

We're going to be editing a file called `/etc/dhcpcd.conf` . What is DHCP I hear you ask. DHCP stands for Dynamic Host Configuration Protocol. This is a network management protocol used to automate the process of configuring devices on IP networks.  A DHCP server dynamically assigns an IP address and other network configuration parameters to each device on a network so they can communicate with other IP networks.

When you first join a network, you, the  client, have no IP address.  In this very early stage of IP communication, you either request to be given dynamically an IP address by the protocol or you tell it a predefined static IP address (which no else must be using at that time, otherwise your request will be rejected). DHCP does all the backend work of reusing IP addresses that haven't been used for a while such that the administrator of the network doesn't' have to think about it.

So let's tell DHCP that we want to predefine our static IP address. Quick side note, we already mentioned that for each network we join we get a different IP address. So you need to indicate which network you're setting up a static address for (i.e.: ethernet/wireless etc)


### Exercise: Setup a static IP address (and cautionary tales!)

1. Configure a Static IP address for your ethernet/WiFi by following these [instructions](https://howchoo.com/pi/configure-static-ip-address-raspberry-pi#find-your-router-ip-address). Let's see what it involves. 

2. Our DHCP (sounds pleasantly familiar now, right?) configuration file needs to be modified
```bash
sudo nano /etc/dhcpcd.conf
```

3. We're going to have to decide on a few things which should now make sense:

```bash
interface <Network>
    static ip_address=<Static_IP>/24
    static routers=<Router_IP>
    static domain_name_servers=<Name_Server>
```
4. Interface selection is simple `eth0` or `wlan0`. Let's setup a static ethernet connection for now (they are obviously not exclusive). 
5. The next step is deciding on a `static ip_address`. What can I choose? This is the annoying part where usually no one tells you and things can go wrong... You should be able to choose whatever IP address you want, however remember that the DHCP system on your network has already allocated IP addresses to other devices etc, so there are constraints. Furthermore, networks typically allocate a range of IP addresses to do certain things. A safe bet is to check what your current ethernet IP address is (that was dynamically allocated). (See section above to do this). If you want to change it (to really feel the control) - then check out what your current IP address is e.g.: `172.24.242.21`, and select something in the same range: e.g.: `172.24.242.200` - this is likely to be a valid IP address. Ideally, we'd find out which IP address in our network are available...  The next two entries should be self-explanatory. 

6. Test it out! (e.g.: set your static IP address not to your current one, disconnect and reconnect to your network and check to see what IP DHCP gave you)

#### References

- https://howchoo.com/pi/configure-static-ip-address-raspberry-pi#find-your-router-ip-address
- https://www.circuitbasics.com/how-to-set-up-a-static-ip-on-the-raspberry-pi/

----

## Domain Name System (DNS) and setting up your brain NB3 as a routed wireless access point

If we want to gain some independence, it might be nice to create our own "network" so that we can connect all of our devices RPi to one same network that we fully control and understand. One way of doing this is to set up your brain NB3 to create a secondary WiFi network. In this network, you alone will be responsible for assigning IP addresses. 

In order to provide network management services (DNS, DHCP) to wireless clients, the Raspberry Pi will need a dnsmasq software package. This package contain a DHCP server (we now know what that does), and amongst other things a DNS (Domain Name System). 

Let's pause to see what that is as it's also a key part of the internet.

We've talked about IP addresses, but most of us never use them. We usually access a web server as: www.someone.com . How does your web browser know where on the Internet this computer lives? DNS is the answer. The DNS is a distributed database which keeps track of computer's names and their corresponding IP addresses on the Internet.

Many computers connected to the Internet host part of the DNS database and the software that allows others to access it (which we're about to get ourselves). These computers are known as DNS servers. No DNS server contains the entire database - they only contain a subset of it. If a DNS server does not contain the domain name requested by another computer, the DNS server re-directs the requesting computer to another DNS server. The Domain Name Service is structured as a hierarchy which makes searching more efficient (e.g.: all the  IP:url allocations associated to e.g.: .com, or .org, or .edu are located in the same subset of computers. Like this, they first look up the domain of the website you've requested e.g.: `.com` and then forward on the request to the right path. 


### Exercise:  Configure your Brain NP3 as a routed wireless access point

1. As we're going to managing a network will need some new packages.
2. We'll be following these [instructions](https://www.raspberrypi.org/documentation/configuration/wireless/access-point-routed.md). Let's make sure we understand each step.
3. As we're going to managing a network will need some new packages: `hostapd` and `dnsmasq`. Hopefully it should make sense now why we need them. 
4. We're going to using the `systemctl` command - this is a utility which is responsible for examining and controlling the systemd system and service manager. (comes up regularly)

5. check out who else is connected to your network using the `nmap` command.  This a free and open-source tool for network discovery that you can download:

```bash
apt install nmap
```
Check out who else is on your network:

```bash
nmap -sn <my-IP-e.g.:192.168.1.0>/24
```




#### References
- https://www.raspberrypi.org/documentation/configuration/wireless/access-point-routed.md

----

## Using the Telegram bot to send your IP address

To be continued
### Exercise:  

More information [in this readme](telegram.md). 

#### References



----
