# Build a Brain : Projects : Drone
This project will use HTML and Python to build a new FPV interface for your NB3.

## Requirements
1. [Connect to your NB3 via SSH using VS Code](/course/versions/buildabrain/_resources/ssh-with-vscode.md)
2. [Open the "LastBlackBox" folder in VS Code](/course/versions/buildabrain/_resources/setup-vscode.md)
3. Synchronize the local LBB repository with GitHub
  - Open a terminal in VS code, run the "Sync" command to update the folder to newest version
    ```bash
    Sync
    ```
  - Close and re-open the VS Code terminal for the changes to take effect

## Create *your* copy of the example project code
- Navigate to the example project folder
```bash
cd /home/$USER/NoBlackBoxes/LastBlackBox/boxes/vision/drone-NB3
```
- Copy the example project Python code to a "my..." version
```bash
cp drone.py my_drone.py
```

## Edit *your* code version in VS Code
- Navigate in VS Code's Explorer to boxes -> vision -> drone-NB3
- Open the **my_drone.py** Python file in the Editor
- Add a new response, such as a sequence of movements, in the "do_action" section
```python
elif command == "do_action":
  # ADD YOUR COMMAND RESPONSES AFTER HERE ------->
  # - What action should your robot do when the "?" is pressed?
  # <------- ADD YOUR COMMAND BEFORE RESPONSES HERE        
```
- A new action sequence coulld look like this...
```python
elif command == "do_action":
  # ADD YOUR COMMAND RESPONSES AFTER HERE ------->
  ser.write(b'l')
  time.sleep(0.1)
  ser.write(b'r')
  time.sleep(0.1)
  ser.write(b'l')
  time.sleep(0.1)
  ser.write(b'r')
  time.sleep(0.1)
  ser.write(b'x')
  # <------- ADD YOUR COMMAND BEFORE RESPONSES HERE        
```
***IMRPORTANT**: When you put your code into the **my_drone.py** file, then it must follow the same indentation (i.e. spacing) as the other sections...otherwise Python will complain!

## Run *your* code version
When you are in the same folder as your python code, then you can run the following command.
```bash
python my_drone.py
```
However, we have also created a "shortcut" that you can run from anywhere.
```bash
MyDrone
```