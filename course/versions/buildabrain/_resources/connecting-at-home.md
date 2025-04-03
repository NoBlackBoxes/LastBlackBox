# Connecting at Home
Do you want to connect to (and develop) your NB3 at home? Here is how.

## Requirements
- A personal computer (laptop or desktop) running Windows, MacOS, or Linux
- A mobile phone capable of creating a WiFi "hotspot"

## Install VS Code on your PC
Follow the instructions [here](https://code.visualstudio.com/) to install VS Code on your home PC.

## Change the name and password of your phone hotspot
Your NB3 robot will connect to *any* WiFi network with the name "LBB" and the password "noblackboxes". To change the name and password of your mobile phone's hotspot network, please try the following:

### iPhone
- *Name*: Go to Settings > General > About > Name, then delete the existing name and type your new one (LBB). Hit “Done” to save your changes.
- *Password*: Go to Settings > Cellular > Personal Hotspot > Wi-Fi Password, then delete the existing password and type your new one (noblackboxes).

### Android 
- Name and Password: Go to Settings > Wireless & Networks > Hotspot & Tethering > WiFi Hotspot, then tap the hotspot name and enter your new one (LBB), then tap the hotspot password and enter your new one (noblackboxes). Hit “Save” to confirm your changes.

## Connect to your mobile hotspot
- Connect your PC to your phone's WiFi hotspot.
- Turn on your NB3. It should connect *automatically* to your phone's LBB network.
- Wait until your phone shows "2 devices connected"

## Find the IP address of your NB3
There are many different software programmes designed to map local networks. I recommend [NMAP](https://nmap.org/download.html), which you can download and install on Linux, MacOS, and Windows. Please follow the install instructions for your Host's OS.

0. Open VS Code on your PC and open a command line (text) "terminal"
1. Determine the "local" IP address of your Host computer

    ```bash
    # Linux or MacOS
    ip address
    # or
    ifconfig

    # Windows
    # From a command line terminal
    ipconfig
    ```
    
    - The IP address of each network device will be listed, if you are connected via WiFi, one device will start with a "w" and the corresponding IP address will *usually* look like this: **192.168.???.???**

2. Map the local network

    ```bash
    nmap 192.168.???.*
    # Use your "host" IP in the command
    ```

    - The output may or may not contain the "hostnames" of connected devices. You are looking for any device with an open port 22 (SSH). There should only be 2 devices at this stage, one is your PC and the other is your NB3!

3. *Alternative*: On MacOS (and some Linux distributions), the command "arp" is pre-installed and can achieve similar results to "nmap". If you have trouble installing NMAP, then try the following in your command terminal:

    ```bash
    arp -a
    ```

## SSH into your NB3
Use VS Code's "Remote-SSH extension" to connect to your NB3. Follow the instructions [here](/course/versions/buildabrain/_resources/ssh-with-vscode.md).

## Congratulations on connecting to your NB3.
...uh...wait, but what if doesn't work? 

What if you want to connect to a different network (e.g. your home's WiFi) rather than always relying on your mobile phone's hotspot? 

Start [here](https://vimeo.com/manage/videos/1036391512). This video explains many different ways to connect to your NB3 (and many common ways that things go wrong). Good luck, and if your are stuck, get in touch! (info@noblackboxes.org)