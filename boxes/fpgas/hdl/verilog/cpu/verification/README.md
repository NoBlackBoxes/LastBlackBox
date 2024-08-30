# Building the RISC-V RV32I toolchain (and libraries)

- clone repo (riscv-gnu-toolchain)
- enter directory

```bash
mkdir build; cd build
../configure --with-arch=rv32i --prefix=/home/${USER}/NoBlackBoxes/tools/rv32i-toolchain
make -j4
```
