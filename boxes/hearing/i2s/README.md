# systems : i2s

Notes: Installing an I2S mems mic on RPI (4b)

- Compiling a kernel module using the "rpi-source" headers for RPIOS 64-bit resulted in "_mcount undefined"
  - This is to do with function tracing/profiling not configured correctly in the compiler kernel or toolchain (unclear)
- There was already a kernel module (ics43432.ko) (for the Innonsense MEMS mic) in linux/sound/soc/codecs (/lib/modules/5.4.51-v8+/kernel/sound/soc/codecs)
- insmod this KO and it works with Alsa (after reboot)

```bash
sudo insmod snd-soc-ics43432.ko
````

- Have to add custom device tree overlay (to overlays and change config.txt)
- It needs tweaks to control volume via software etc.
