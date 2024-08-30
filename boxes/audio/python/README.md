# Audio : python

```bash
pip install --upgrade pip setuptools wheel
sudo apt install python3-dev build-essential
sudo apt install portaudio19-dev # required for 64-bit pyaudio build
pip install pyaudio
```

```bash
# On Host (current Python version 3.12.3)
echo "/home/${USER}/NoBlackBoxes/LastBlackBox/boxes/audio/python/libs" > /home/${USER}/NoBlackBoxes/LastBlackBox/_tmp/NBB/lib/python3.12/site-packages/NBB_sound.pth
```

## Profiling

PC:
Input callback, 48000 rate, 4800 buffer, int16, 1 channel, Float32 internal conversion: Avg (Max) Callback Duration (us): 108.35 (1882.55)
Input callback, 48000 rate, 4800 buffer, int16, 2 channel, Float32 internal conversion: Avg (Max) Callback Duration (us): 171.05 (3262.52)
Input callback, 48000 rate, 4800 buffer, int32, 2 channel, Float32 internal conversion: Avg (Max) Callback Duration (us): 267.15 (3635.64)
Output callback,44100 rate, 4410 buffer, int16, 2 channel, Float32 internal conversion: Avg (Max) Callback Duration (us): 37.67 (99.18)

RPi:
Input callback, 48000 rate, 4800 buffer, int32, 1 channel, Float32 internal conversion: Avg (Max) Callback Duration (us): 864.54 (7222.89)
Input callback, 48000 rate, 4800 buffer, int32, 2 channel, Float32 internal conversion: Avg (Max) Callback Duration (us): 1704.76 (31641.96)
Output callback,44100 rate, 4410 buffer, int16, 2 channel, Float32 internal conversion: Avg (Max) Callback Duration (us): 188.47 (483.75)

- Turn off throttling/govenor

```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
sudo sh -c "echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
```

Input callback, 48000 rate, 4800 buffer, int32, 2 channel, Float32 multiply conversion: Avg (Max) Callback Duration (us): 829.71 (7487.30)
Input callback, 48000 rate, 4800 buffer, int32, 2 channel, Float32 division conversion: Avg (Max) Callback Duration (us): 1739.91 (14403.82)