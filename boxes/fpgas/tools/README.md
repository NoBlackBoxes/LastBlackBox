# fpgas : tools

Instaling tools to synthesize and upload to an FPGA

## Create/activate python virtual environment
```bash
export NBB_ROOT=/home/kampff/NoBlackBoxes/
cd $NBB_ROOT/tools/environments
python -m venv nbb-python
source nbb-python/bin/activate

export NBB_ROOT=/home/kampff/NoBlackBoxes/
source $NBB_ROOT/tools/environments/nbb-python/bin/activate

```

## Notes for TinyFPGA BX

- Install APIO and tinyprog python tools

```bash
pip install apio tinyprog

# For Upduino3...might need the latest apio
pip install -U apio

apio install system scons ice40 iverilog yosys
apio drivers --serial-enable

# Add user to dialout (not sure for Upduino 3.0)
sudo usermod -a -G dialout $USER # Debian
sudo usermod -a -G uucp $USER # Arch

```



- Update bootloader

```bash
tinyprog --update-bootloader
```

