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
- ???
```python
???
```

## Run *your* code version
When you are in the same folder as your python code, then you can run the following command.
```bash
python my_drone.py
```
However, we have also created a "shortcut" that you can run from anywhere.
```bash
MyDrone
```