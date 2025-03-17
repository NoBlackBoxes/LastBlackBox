# Build a Brain : Projects : Listen
This project will use Python to expand the repertoire of commands that your NB3 understands (and to control the responses).

## Requirements
- Connect to your NB3 via SSH using VS Code
- Open the "LastBlackBox" folder in VS Code
- Synchronize the local repository with GitHub
```bash
Sync
```
- Close and re-open the VS Code terminal

## Create *your* copy of the example project code
- Navigate to the example project folder
```bash
cd /home/$USER/NoBlackBoxes/LastBlackBox/boxes/intelligence/NPU/listen-NB3
```
- Copy the example project Python code to a "my..." version
```bash
cp listen.py my_listen.py
```

## Edit *your* code version in VS Code
- Navigate in VS Code's Explorer to boxes -> intelligence -> NPU -> listen-NB3
- Open the **my_listen.py** Python file in the Editor
- Find the code section where you will add *new* responses for *new* voice commands
```python
# ADD YOUR COMMAND RESPONSES AFTER HERE ------->
if labels[top_3_indices[0]] == "turn_left":  # If the "best" voice command detected is "turn_left"
    ser.write(b'l')                          # Send the Arduino 'l' (the command to start turing left)  
    time.sleep(1.0)                          # Wait (while moving) for 1 second
    ser.write(b'x')                          # Send the Arduino 'x' (the command to stop)
# <------- ADD YOUR COMMAND BEFORE RESPONSES HERE
```
- The "turn_left" response is provided as an example.

## Run *your* code version
When you are in the same folder as your python code, then you can run the following command.
```bash
python my_listen.py
```
However, we have also created a "shortcut" that you can run from anywhere.
```bash
MyListen
```