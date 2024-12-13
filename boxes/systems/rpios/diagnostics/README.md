# systems : rpios : diagnostics
Check the system stats and diagnose any hardware/software "health issues".

## Installation
Install the required packages

```bash
sudo apt install hdparm sysbench speedtest-cli
```

## Sample Output (Raspberry Pi 4b - 2GB, a.k.a. *NB3 Midbrain*)

```txt
Raspberry Pi Diagnostics
------------------------

System Information:
-------------------
Uptime     : 0 days, 3 hours, 40 minutes
Processor  : Cortex-A72
CPU cores  : 4 @ 1800.0000 MHz
RAM        : 1.8 GiB
Swap       : 512.0 MiB
Disk       : 14.4 GiB
Distro     : Debian GNU/Linux 12 (bookworm)
Kernel     : 6.6.51+rpt-rpi-v8
VM Type    : NONE
IPv4/IPv6  : ✔ Online / ❌ Offline

temp=38.4'C
arm_freq=1800
core_freq=500
core_freq_min=200
gpu_freq=500
gpu_freq_min=250
sd_clock=50.000 MHz

Running Internet Speed test...
Ping: 15.008 ms
Download: 95.97 Mbit/s
Upload: 30.79 Mbit/s

Running CPU test (4 cores)...
 total time: 10.0005s
 min: 0.56
 avg: 0.56
 max: 1.07

temp=48.7'C

Running THREADS test (4 cores)...
 total time: 10.0034s
 min: 4.26
 avg: 4.54
 max: 17.50

temp=52.5'C

Running MEMORY test...
3072.00 MiB transferred (8408.44 MiB/sec)
 total time: 0.3619s
 min: 0.00
 avg: 0.00
 max: 4.04

temp=53.0'C

Running SD Card (hdparm) test...
 Timing buffered disk reads: 132 MB in  3.04 seconds =  43.37 MB/sec

temp=44.8'C

Running DD WRITE test...
536870912 bytes (537 MB, 512 MiB) copied, 26.878 s, 20.0 MB/s

temp=42.8'C

Running DD READ test...
536870912 bytes (537 MB, 512 MiB) copied, 11.8649 s, 45.2 MB/s

temp=43.3'C

------------------------
```