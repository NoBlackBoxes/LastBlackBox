# Systems : Programming : Python : Eyes
Example code for controlling the NB3's eyes (8x8 LED matrices)

*Assume right eye CS on CE0 and left eye CS on CE1*

## Prerequisites
We will use the Python spidev (SPI device) library

```bash
# The library should be installed by default on RPiOS, if not, then you can pip install it.
pip install spidev
```
## Add NB3 Eye Python library to LBB environment path
```bash
# Insert the path (first bit of text) into (>) a *.pth file contained in your LBB virtual environment

# On Host (current Python version 3.13.1)
echo "/home/${USER}/NoBlackBoxes/LastBlackBox/boxes/systems/eyes/python/libs" >> /home/${USER}/NoBlackBoxes/LastBlackBox/_tmp/LBB/lib/python3.13/site-packages/NB3.pth

# On NB3 (current Python version 3.11.2)
echo "/home/${USER}/NoBlackBoxes/LastBlackBox/boxes/systems/eyes/python/libs" >> /home/${USER}/NoBlackBoxes/LastBlackBox/_tmp/LBB/lib/python3.11/site-packages/NB3.pth
```

## Example Code

## Wiring Connections

NB3 Eyes PCB
- DIN: Connect lD to rD, Connect rD to MOSI (pin 19)
- CLK: Connect lE to rE, Connect rE to SCLK (pin 23)
- CS: Connect rf to CE0 (pin 24). Connect lF to CE1 (pin 25) 
- 