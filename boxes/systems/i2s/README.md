# systems : I2s

Notes: Installing an I2S mems mic on RPI (4b)

- Compiling a kernel module using the "rpi-source" headers for RPIOS 64-bit resulted in "_mcount undefined"
  - This is to do with function tracing/profiling not configured correctly in the compiler kernel or toolchain (unclear)
- There was already a kernel module (IC44385.ko) (not sure of name) in linux/sound/soc
- insmod this KO and it works with Alsa (after reboot)
- It needs tweaks to control volume via software etc.
