# vision

Look at everyone.

----

<details><summary><b>Materials</b></summary><p>

Contents|Description| # |Data|Link|
:-------|:----------|:-:|:--:|:--:|
Camera (RPiv2)|Raspberry Pi 8 MP camera|1|[-D-](_data/datasheets/rpi_camera_v2.pdf)|[-L-](https://uk.farnell.com/raspberry-pi/rpi-8mp-camera-board/raspberry-pi-camera-board-v2/dp/2510728)
Mount|Custom laser cut camera mount|1|-|-
M2 screw (12)| 12 mm long M1 screw with phillips socket|4|-|-
M2 nut| M2 nut|4|-|-
M2 washer|nylon washer 2-3 mm inner diameter|4|-|-
M3 screw (20)| 20 mm long M3 screw with phillips socket|2|-|-
M3 nut| M3 nut|4|-|-

Required|Description| # |Box|
:-------|:----------|:-:|:-:|
Multimeter|(Sealy MM18) pocket digital multimeter|1|[white](/boxes/white/README.md)|

</p></details>

----

## NB3

This box will contribute the following (red) components to your NB3

<p align="center">
<img src="_images/NB3_vision.png" alt="NB3 stage" width="400" height="400">
<p>

----


## Goals

### Concepts
- cameras
- CCDs
- drivers
- kernels and lodable kernel modules

### Tasks and challenges
- connect camera to NB3 and look at the world
- broadcast what you see to your local network using a grey box
- broadcast what you see to your local network using a white box
- access video content remotely

----  

## Cameras

### Exercise: Connect camera to NB3

1. Shutdown (and power off) your NB3 RPi
1. Install your camera mount
 - Attach camera to mount (use the nylon washers in the upper two mounting screws to keep the board level when fixed)
 - Attach mount to NB3 using long M3 screws and nuts
2. Connect your camera to your RPi board (the blue strips on the flat ribbon cable should face towards the USB port on the RPi board and away from the lens on the camera board)
3. Turn your NB3 RPi back on


### Exercise: Preview content of camera

----  

## Live streaming


Camera drivers... Loadable kernels...

### Exercise: Live stream camera data to local network (using a black box)

We're first going to broadcast our video using a black box (not optimal - but a quick way to see if everything is working). This exercise might be removed in the future but it allows us to cover some interesting things. 

1. One good set off instructions can be found [here](https://raspberrytips.com/raspberry-pi-camera-as-webcam/). Let's unpack what we have to do.
2. First install some packages (after making sure that our camera is activated):

```bash
sudo apt install motion libavcodec-dev libavformat-dev libavutil-dev libjpeg-dev libjpeg62-turbo-dev libpq-dev libswresample-dev
```

3. Enable the camera driver:
```bash
sudo modprobe bcm2835-v4l2
```
What is `modeprobe`? Modprobe is a Linux program used to add or remove a loadable kernel module to the Linux kernel. What is a kernel and what is a loadable kernel? To get a comprehensive understanding, check out this interesting [site] (https://linux.die.net/lkmpg/x40.html). More details to be added. 

4. The rest of the instructions should be self explanatory. 
5. Check out your video stream at `http://<RASPBERRY_IP>:8081`!
6. The image might need to be rotated - edit your motion configuration file:
```bash
rotate 180
```
7. Close the application using `ctrl + c` 

### Exercise: Live stream camera data to local network (using a white box)

Now let's do it the right way - tutorial was based on this [example](https://randomnerdtutorials.com/video-streaming-with-raspberry-pi-camera/).
Check out the video-stream folder for more details about the exercise.


### References
- https://raspberrytips.com/raspberry-pi-camera-as-webcam/
- https://linux.die.net/lkmpg/x40.html
- https://randomnerdtutorials.com/video-streaming-with-raspberry-pi-camera/

---- 



----  

## CCDs

And Bayer....

### Exercise: See


### References

Good luck!!! :)

----
