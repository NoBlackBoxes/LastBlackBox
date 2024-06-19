# The Last Black Box: *Build a Brain*

## Project: Build the robot's body

Let's assemble a "body" to attach a muscles (motors), eyes (sensors), and a brain (Arduino)

### Step 1: Attach your "hindbrain" breadboard

Remove the yellow paper on the bottom to expose the adhesive back and stick your breadboard to the robot body. 
- ***Note***: If you have circuits on your breadboard already, then you can leave them in place.
- ***Note***: Make sure your USB connection points to the *BACK* of the robot.

<p align="center">
<img src="../../resources/images/1_NB3_body_hindbrain_install.png" alt="1_NB3_body_hindbrain_install" width="350" height="330">
</p>

### Step 2: Mount your motors

1. Get the following screws and standoffs

<p align="center">
<img src="../../resources/images/2_servo_mounting_screws_A.png" alt="2_servo_mounting_screws_A" width="400" height="200">
</p>


2. Install them in the body

<p align="center">
<img src="../../resources/images/3_servo_mounting_example_A.png" alt="3_servo_mounting_example_A" width="450" height="330">
</p>


2. Get two servo motors, more screws, and the plastic mounting brackets

<p align="center">
<img src="../../resources/images/4_servos_and_mounting_hardware_B.png" alt="4_servos_and_mounting_hardware_B" width="450" height="300">
</p>

3. Fix the motors in place with the screws and mounting brackets

<p align="center">
<img src="../../resources/images/6_servos_mounted_B.png" alt="6_servos_mounted_B" width="400" height="300">
</p>

<p align="center">
<img src="../../resources/images/7_servos_mounted_C.png" alt="7_servos_mounted_C" width="400" height="300">
</p>

- ***NOTE***: Pay attention to the orientation of the left and right motors

<p align="center">
<img src="../../resources/images/5_servos_mounted_A.png" alt="5_servos_mounted_A" width="650" height="300">
</p>

### Step 3: Mount your front ball bearing

1. Get the ball caster bag

<p align="center">
<img src="../../resources/images/8_caster_package.png" alt="8_caster_package" width="400" height="400">
</p>


2. Remove the ball (use your screwdriver to pop it out)

<p align="center">
<img src="../../resources/images/9_caster_removed.png" alt="9_caster_removed" width="400" height="300">
</p>

3. Insert the shorter screws through the caster base and the two disc spacers

<p align="center">
<img src="../../resources/images/10_caster_spacer_screws.png" alt="10_caster_spacer_screws" width="400" height="300">
</p>

<p align="center">
<img src="../../resources/images/11_caster_scews_top_A.png" alt="11_caster_scews_top_A" width="400" height="300">
</p>

4. Insert the screws into the robot body and use the nuts to fix them in place

<p align="center">
<img src="../../resources/images/13_caster_nuts.png" alt="13_caster_nuts" width="500" height="250">
</p>

<p align="center">
<img src="../../resources/images/12_caster_screws_top_B.png" alt="12_caster_screws_top_B" width="400" height="200">
</p>

<p align="center">
<img src="../../resources/images/14_caster_mounted.png" alt="14_caster_mounted" width="400" height="400">
</p>

5. Put the ball back in the holder

<p align="center">
<img src="../../resources/images/15_caster_ball_installed.png" alt="15_caster_ball_installed" width="400" height="400">
</p>

### Step 4: Mount your wheels

1. Get the two wheel bags

<p align="center">
<img src="../../resources/images/16_wheeel_packages.png" alt="16_wheeel_packages" width="500" height="400">
</p>

2. Open and remove the wheels AND the small screws

<p align="center">
<img src="../../resources/images/17_wheel_mounting_hardware.png" alt="17_wheel_mounting_hardware" width="550" height="400">
</p>

3. Stretch the rubber over the outside of the wheel

<p align="center">
<img src="../../resources/images/18_wheel_rubber_installed.png" alt="18_wheel_rubber_installed" width="550" height="320">
</p>

4. Find the mounting whole in each wheel

<p align="center">
<img src="../../resources/images/19_wheel_mounting_hole.png" alt="19_wheel_mounting_hole" width="550" height="320">
</p>

5. Find the mounting cylinder on each motor

<p align="center">
<img src="../../resources/images/20_servo_mounting_pin.png" alt="20_servo_mounting_pin" width="550" height="320">
</p>

6. Push the wheel mounting hole onto to the motor's mounting cylinder

<p align="center">
<img src="../../resources/images/21_wheel_mounted.png" alt="21_wheel_mounted" width="550" height="450">
</p>

7. Use each of the small screws to fix the wheel to the motor

<p align="center">
<img src="../../resources/images/22_wheel_mounted_screw.png" alt="22_wheel_mounted_screw" width="550" height="450">
</p>

### Step 5: Connect the motor cables to the robot body

1. The right motor cable connects with the orange wire at **RA**

<p align="center">
<img src="../../resources/images/23_right_servo_wires.png" alt="23_right_servo_wires" width="500" height="400">
</p>


2. The left motor cable connects with the orange wire at **LA**

<p align="center">
<img src="../../resources/images/24_left_servo_wires.png" alt="24_left_servo_wires" width="500" height="400">
</p>


### Step 6: Add power connections for your motors and Arduino

1. Connect the batteries to the breadboard

<p align="center">
<img src="../../resources/images/25_battery_power_breadboard.png" alt="25_battery_power_breadboard" width="700" height="400">
</p>

2. Find your jumper wires

<p align="center">
<img src="../../resources/images/26_jumper_wires.png" alt="26_jumper_wires" width="700" height="300">
</p>

2. Use jumpers to connect your breadboard to the robot body, red lane to 5V, blue lane to 0V

<p align="center">
<img src="../../resources/images/27_body_power.png" alt="27_body_power" width="750" height="500">
</p>

3. Use jumpers to connect your Arduino to the breadboard power

<p align="center">
<img src="../../resources/images/28_arduino_power.png" alt="28_arduino_power" width="750" height="600">
</p>

- ***NOTE***: Use a long red jumper to connect the red lane of the jumper (+) to the other side. See image below!

<p align="center">
<img src="../../resources/images/29_cross_power.png" alt="29_cross_power" width="750" height="500">
</p>

### Step 7: Connect the Arduino to your motors

1. Use a longer cable to connect Pin 9 (the 3rd pin from the bottom left) on the Arduino to **LA** (for the left motor)

<p align="center">
<img src="../../resources/images/30_left_servo_control.png" alt="30_left_servo_control" width="750" height="500">
</p>

2. Use a longer cable to connect Pin 10 (the 4th pin from the bottom left) on the Arduino to **RA** (for the right motor)

<p align="center">
<img src="../../resources/images/31_right_servo_control.png" alt="31_right_servo_control" width="750" height="500">
</p>

### Step 8: Test your motors

1. If you switch on your batteries, then your motors should both "twitch". 

### Step 9: Control your motors

1. Upload the following Arduino code to your board using your USB cable.

```c++
#include <Servo.h>  // This includes the "servo" library

Servo left, right;  // This creates to servo objects, one for each motor

int speed = 0;      // This creates a variable called "speed" that is intially set to 0

// Setup
void setup() {
  right.attach(9);  // Assign right servo to digital (PWM) pin 9 (change accorinding to your connection)
  left.attach(10);  // Assign left servo to digital (PWM) pin 10 (change accorinding to your connection)
}

void loop() {

  // Servos are often used to control "angle" of the motor, therefore the "servo library" uses a range of 0 to 180.
  // Your servos control "speed", therefore 0 is full speed clockwise, 90 is stopped, and 180 is full speed counter-clockwise

  // Move left servo through the full range of speeds
  for (speed = 0; speed <= 180; speed += 1) {
    left.write(speed);
    delay(15);
  }
  left.write(90); // stop left servo
  
  // Move right servo
  for (speed = 0; speed <= 180; speed += 1) {
    right.write(speed);
    delay(15);
  }
  right.write(90); // stop right servo
}
```

***Have Fun!***