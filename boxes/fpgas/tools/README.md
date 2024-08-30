# fpgas : tools

Instaling tools to synthesize and upload to an FPGA

## Create/activate python virtual environment
```bash
export NBB_ROOT=/home/${USER}/NoBlackBoxes/
cd $NBB_ROOT/tools/environments
python -m venv nbb-python
source nbb-python/bin/activate

export NBB_ROOT=/home/${USER}/NoBlackBoxes/
source $NBB_ROOT/tools/environments/nbb-python/bin/activate

```

### TO DO

Need to add NB3_hindbrain to board file in APIO lib (site-packages)
Need correct FTDI descriptor (can I change it? EEPROM?)
Need to generate an apio.ini file in synthesis...

```json
  "NB3_hindbrain": {
    "name": "NB3 Hindbrain",
    "fpga": "iCE40-UP5K-SG48",
    "programmer": {
      "type": "iceprog"
    },
    "usb": {
      "vid": "0403",
      "pid": "6010"
    },
    "ftdi": {
      "desc": "Dual RS232-HS"
    }
  },
```

## Notes for Upduino3

### Linux

- Install APIO

```bash
pip install apio

# For NB3_hindbrain...might need the latest apio
pip install -U apio

apio install system scons ice40 iverilog yosys
apio drivers --serial-enable

# Add user to dialout (not sure for Upduino 3.0)
sudo usermod -a -G dialout $USER # Debian
sudo usermod -a -G uucp $USER # Arch

```

- Update UDEV rules in /etc/udev

### MacOS

- brew install libffi libftdi
- bew install apio

### Windows

- Install WSL version 2
- Install APIO
- Use "iceprog.exe" from windows command prompt for upload
