# fpgas : hdl : verilog

We will use the Verilog hardware description language (HDL) to design digital logic devices

## Prerequisites

Verilog files describe hardware. They don't "do" anything by themselves. You must use a "simulator" to simulate their behaviour or a "synthesizer" to synthesize a hardware design that can be run on an FPGA (or used to create a silicon device)

### Install a Verilog simulator

We will use the Icarus open-source Verilog simulator

```bash
sudo apt-get install iverilog
```

We will also use an open-source visualizer (GTKWave) to view the results of our hardware simulations

```bash
sudo apt-get install gtkwave
```

### Install a Synthesis toolchain

We will use an (amazing) open source sythesis (yosys) and FPGA "programming" toolchain (IceStorm) developed by reverse-engineering the proprietary "bitstream" formats of an FPGA vendor (Lattice). *Note: Lattice has been quite supportive of these efforts. Kudos*

### Install WaveTrace (VScode extension) to view VCD files


### Custom Board (NB3 Hindrbain)

Add to API0 boards.json

```txt
  "NB3_hindbrain": {
    "name": "NB3 Hindbrain v0",
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