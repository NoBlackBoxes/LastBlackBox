# fpgas : tools

Instaling tools to synthesize and upload to an FPGA

## Create/activate python virtual environment
```bash
export LBBROOT=/home/kampff/NoBlackBoxes/
cd $LBBROOT/tools/environments
python -m venv lbb-python
source lbb-python/bin/activate
```

## Notes fore TinyFPGA BX

- Install APIO and tinyprog python tools

```bash
pip install apio==0.4.0b5 tinyprog
apio install system scons icestorm iverilog
apio drivers --serial-enable

# Add user to dialout
sudo usermod -a -G dialout $USER # Debian
sudo usermod -a -G uucp $USER # Arch
```

- Update bootloader

```bash
tinyprog --update-bootloader
```