# Connecting your RPi to your Arduino

- Install Arduino command line utilities (add to $PATH or copy to /usr/bin)
 -  Download Linux ARM 32-bit version (tar.gzipped archive), (secure)copy (scp) to RPi and extract
- Install arduino-cli core arduino:avr
- Compile: arduino-cli compile --fqbn arduino:avr:nano
- Upload: arduino-cli upload -p /dev/ttyUSB0 -fqbn arduino:avr:nano (name of sketch)

- *NOTE*: Sketch (*.ino) must have the same name as the folder it is in.