# Motors : DC Motors
An electric motor converts current into rotation using electromagnets that are turned on and off in a coordinated pattern. Different types of motors (stepper, brushed, or brushless) use different strategies (circuits) for this coordination.

## [Video](https://vimeo.com/1031627739)

## Concepts
- We know we can generate a magnetic force using an electromagnet. How do we turn that push-pull force into a rotational force...spin?
- Ideas from compass...multiple coils around a bar magnet on an axle.
- We need to activate the coils in the correct sequence.
- This is how a stepper motor works...but they are more complex to control, and although they can be precisely rotated, often have limited max speed.
- We would like a simpler design.
- For this we will use something called a brushed DC motor.
- Here we place the heavier magnets on the outside, one N amd S.
- In the simplest bi-polar motor, we have two coils, wound in opposite directions.
- A brush drags along the axle, sending current through the coils, and flipping the direction of the current and thus the poles of the electromagnets to continue the direction of rotation.
- We only need two wires.
- However, there is an ambiguity if such a motor starts in a perfectly aligned position. Which way does it rotate first?
- Most brushed DC motors, such as the ones in your kit, use three coils to break this ambiguity. It will also allow us to control the direction that the motor spins by swapping the direction we send current through the motor.
- Let's try this.
- Finally, motors have mass. They don't stop as soon as the applied voltage is disconnected.
- The coils will continue to rotate, moving through the magnetic field of the permanent magnets...which will induce a current in the coils.
- You must be careful to deal with this induced EMF...as it can damage sensitive electronics connected ot the motor circuit...but we will let you know when this could happen.
- This inertial EMF, and the current it can generate can also be useful. This is similar to how an electrical generator works, and how some electric cars recharge their batteries while braking.
- You can see this effect by connecting one motor to another...and using one as a generator to move the other.

## Lesson

- **TASK**: Play with your brushed DC motor. Spin it forwards *and* backwards...
    - *Challenge*: What are some ways you could change the *speed* with which your motor spins?
> Switching the direction that current flows through your motor will change the direction it spins.
