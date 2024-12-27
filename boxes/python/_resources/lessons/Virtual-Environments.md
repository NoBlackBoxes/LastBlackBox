# Python : Virtual Environments
We will next create a Python **virtual environment** on our NB3 that will isolate the specific Python packages we require for the course from the Python packages used by the Raspberry Pi's operating system.

## [Video]()

## Concepts
- Many different pieces of software on your operating system use Python (and associated libraries).
- Often, programs will require a specific version of a library/package to run...or even a specific version of Python. This can be true of both system software and projects you find on the internet.
- A "virtual environment" allows us to create separate spaces for different Python installations, which can help prevent version conflicts with our system or between projects.
- On our NB3, we will create a virtual environment for the LBB course that includes the "system packages" of the OS. This will give us access to many useful libraries, but will prevent any new libraries we install from messing up the OS.

## Connections

## Lesson

- **TASK**: Create a "virtual environment" called LBB
    - Follow the instructions here: [virtual environments](../../../boxes/python/virtual_environments/README.md)
> You should now have a virtual environment activated (and installed in the folder "_tmp/LBB").