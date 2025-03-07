# Systems : Programming : Python : Eyes
Example code for controlling the NB3's eyes (8x8 LED matrices)

*Assume right eye CS on CE0 and left eye CS on CE1*

## Prerequisites
We will use the Python spidev (SPI device) library

```bash
# The library should be installed by default on RPiOS, if not, then you can pip install it.
pip install spidev
```

## Example Code

## Wiring Connections

NB3 Eyes PCB
- DIN: Connect lD to rD, Connect rD to MOSI (pin 19)
- CLK: Connect lE to rE, Connect rE to SCLK (pin 23)
- CS: Connect rf to CE0 (pin 24). Connect lF to CE1 (pin 25) 
- 