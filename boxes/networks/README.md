# networks

Talk to everyone.

----

<details><summary><b>Materials</b></summary><p>

Contents|Description| # |Data|Link|
:-------|:----------|:-:|:--:|:--:|
Cable (Ethernet)|RJ45 cact5e ethernet patch cable (1 m)|1|[-D-](_data/datasheets/ethernet_cable_1m.pdf)|[-L-](https://uk.farnell.com/pro-signal/ps11074/lead-patch-cat-5e-1-00m-black/dp/1734943)

Required|Description| # |Box|
:-------|:----------|:-:|:-:|
Multimeter|(Sealy MM18) pocket digital multimeter|1|[white](/boxes/white/README.md)|

</p></details>

----

## Connect to your NB3 RPi over a "home" WiFi network (i.e. a WiFi router you control)

We will use "ssh" to connect to our NB3 RPi from our LBB host computer.


*From your NB3...*

1. Determine the IP address of your NB3 RPi

```bash
ifconfig
```

2. Create SHA keys

```bash
sudo dpkg-reconfigure openssh-server
```

*From your LBB host computer...*

1. Connect via SSH

```bash
ssh student@<IP-ADDRESS-OF-NB3>
```

## Protocols

Learn the language

### Exercise: Connect

Good luck!!! :)

----
