# Build a Brain : Projects : Look
This project will use Python to control how your NB3 responds to detected faces.

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
cd $HOME/NoBlackBoxes/LastBlackBox/boxes/intelligence/NPU/look-NB3
```
- Copy the example project Python code to a "my..." version
```bash
cp look.py my_look.py
```

## Edit *your* code version in VS Code
- Navigate in VS Code's Explorer to boxes -> intelligence -> NPU -> look-NB3
- Open the **my_look.py** Python file in the Editor
- Find the code sections where you can add *your* responses to detected faces
```python
if num_faces == 0:          # If NO face detected
    # ADD YOUR COMMAND RESPONSES AFTER HERE ------->
    ser.write(b'x')                          # Send the Arduino 'x' (the command to stop)
    # <------- ADD YOUR COMMAND BEFORE RESPONSES HERE
```
- If no faces are detected, then the robot is told to stop moving
```python
elif num_faces == 1:        # If ONE face detected
    # Process ONE face
    rectangle, x, y = process_face(output_rects[0])

    # ADD YOUR COMMAND RESPONSES AFTER HERE ------->
    ser.write(b'x')                          # Send the Arduino 'x' (the command to stop)
    # <------- ADD YOUR COMMAND BEFORE RESPONSES HERE
```
- If one face is detected, then the robot is again told to stop moving. However, this time the robot knows the X (horizontal) and Y (vertical) position of the detected face. You could decide to have the robot turn *towards* the face and thus "track" the person it sees. How would you do this? 
  - *Hint*: The x variable tells you the position of the the detected face. 0.0 is a face all the way to the left of the image, 1.0 is a face all the way to the right, and 0.5 is a face directly in the middle of the image.

```python
elif num_faces >= 2:         # If TWO (or more) faces detected
    # Process TWO faces
    rectangle1, x1, y1 = process_face(output_rects[0])
    rectangle2, x2, y2 = process_face(output_rects[1])

    # ADD YOUR COMMAND RESPONSES AFTER HERE ------->
    ser.write(b'x')                          # Send the Arduino 'x' (the command to stop)
    # <------- ADD YOUR COMMAND BEFORE RESPONSES HERE
```
- If two faces are detected, then the robot is *AGAIN** told to stop moving. This is pretty boring...so it is up to you to decide how the robot responds when **two** faces are detected! You will have the positions of both faces (e.g. x1 and x2), but be aware...the neural network does not *recognise* faces it only tells you that there is a face (or two) in this part of the image, not who it is. Therefore, face #1 and face #2 might swap labels in the next image taken by the camera.

***IMRPORTANT**: When you put your code into the **my_look.py** file, then it must follow the same indentation (i.e. spacing) as the other sections...otherwise Python will complain!

## Run *your* code version
When you are in the same folder as your python code, then you can run the following command.
```bash
python my_look.py
```
However, we have also created a "shortcut" that you can run from anywhere.
```bash
MyLook
```