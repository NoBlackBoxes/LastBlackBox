# Linux
A free and open source operating system.

## [Introduction](https://vimeo.com/1005196173)
> Linux is based on UNIX.

- **Task**: Explore Linux. Spend any extra time you have fiddling, playing with the UNIX approach to controlling a computer. Create some folders. Edit some files.
  - Some other task notes
  > Expected results from playing with Linux

## [Git](https://vimeo.com/??????)
> Git is a program that keeps track of changes to your files. It is very useful when developing code. This entire course is stores as a git "repository".

- **Task**: "Clone" (copy) all of the code in the [LastBlackBox](https://github.com/NoBlackBoxes/LastBlackBox) GitHub repository directly to your NB3's midbrain. 
  - *Help*: It will help with later exercises if you put this example code in a specific location on the Raspberry Pi.
  - Run these commands from your user's home folder:
  - [Code]
    ```bash
    cd ~                # Navigate to "home" directory
    mkdir NoBlackBoxes  # Create NoBlackBoxes directory
    cd NoBlackBoxes     # Change to NoBlackBoxes directory

    # Clone LBB repo (only the most recent version)
    git clone --depth 1 https://github.com/NoBlackBoxes/LastBlackBox
    ```
  > You should now have a folder called "LastBlackBox" (inside the "NoBlackBoxes" folder) that contains the exact same files as the online GitHub repo.
