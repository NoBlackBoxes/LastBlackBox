# Python : Virtual Environments
We will next create a Python **virtual environment** on our NB3 that will isolate the specific Python packages we require for the course from the Python packages used by the Raspberry Pi's operating system.

## [Video](https://vimeo.com/1042637566)

## Concepts
- Many different pieces of software on your operating system use Python (and associated libraries).
- Often, programs will require a specific version of a library/package to run...or even a specific version of Python. This can be true of both system software and projects you find on the internet.
- A "virtual environment" allows us to create separate spaces for different Python installations, which can help prevent version conflicts with our system or between projects.
- On our NB3, we will create a virtual environment for the LBB course that includes the "system packages" of the OS. This will give us access to many useful libraries, but will prevent any new libraries we install from messing up the OS.
- We will also need to tell our Python environment where custom libraries, such as those for our NB3, are located. We need to tell it which "paths" to look in. We can do this by adding a *.pth file with the path(s) to our "site-packages" folder.
- We will be using PIP inside of a Virtual Environment...you may have already seen it in action.
- We can view the installed packages (within our current environment) using: pip list
- We may sometimes need to install specific versions of a package, you can do this as well

## Connections

## Lesson

- **TASK**: Create a "virtual environment" called LBB
    - Follow the instructions here: [virtual environments](../../../boxes/python/virtual_environments/README.md)
> You should now have a virtual environment activated (and installed in the folder "_tmp/LBB").

- **TASK**: Install some useful packages using PIP
    - Install numpy
    - Install matplotlib
    - Make a cool plot and save it to an image file
> You should now hav an image of your plot saved, which you can open and view inside VS code.