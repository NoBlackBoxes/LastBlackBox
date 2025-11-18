# Build a Brain : Projects : Listen
This project will use Python to expand the repertoire of commands that your NB3 understands (and to control the responses).

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
cd $HOME/NoBlackBoxes/LastBlackBox/boxes/intelligence/NPU/listen-NB3
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
if best_voice_command == "turn_left":  # If the "best" voice command detected is "turn_left"
    ser.write(b'l')                    # Send the Arduino 'l' (the command to start turing left)  
    time.sleep(1.0)                    # Wait (while moving) for 1 second
    ser.write(b'x')                    # Send the Arduino 'x' (the command to stop)
# <------- ADD YOUR COMMAND BEFORE RESPONSES HERE
```
- The "turn_left" response is provided as an example.

***IMRPORTANT**: When you put your code into the **my_listen.py** file, then it must follow the same indentation (i.e. spacing) as the other sections...otherwise Python will complain!

## Run *your* code version
When you are in the same folder as your python code, then you can run the following command.
```bash
python my_listen.py
```
However, we have also created a "shortcut" that you can run from anywhere.
```bash
MyListen
```

## Ideas
The list of 148 commands that your NB3 understands is [here](/boxes/intelligence/NPU/listen-NB3/model/labels.txt). As you will see, many of the commands make sense for controlling the movement of your robot. We suggest starting with these to create a robot that can navigate an obstacle course just by following your verbal instructions.

However, the NB3 could respond in other ways as well. 

## Playing sounds on command
You could have your NB3 respond by playing a sound effect or song. However, the first step will be to "download" the sound file to your robot!

### Download and install the YouTube download tool.
```bash
mkdir ~/.local/bin
wget https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -O ~/.local/bin/yt-dlp
chmod a+rx ~/.local/bin/yt-dlp  # Make executable
```

### Navigate to your "Sounds" folder
You should download all your sound files to the same folder (the "temporary sounds" folder). You can do this by first navigating to that folder in the Terminal window.
```bash
cd $LBB/_tmp/sounds
```

### Download your favourite audio from YouTube
Remember to change the URL (http://...) to your music/sound effect
```bash
 ~/.local/bin/yt-dlp -f bestaudio --extract-audio --audio-format wav --audio-quality 16K --postprocessor-args "-ar 16000" -o my_song.wav https://www.youtube.com/watch?v=dQw4w9WgXcQ
```
If the sound you download sounds "funny", too slow or too fast, then you can use the command below to *convert* it to the correct speed.
```bash
# Convert the sample rate to 16 kHz, which is what the "Listen" demo needs
ffmpeg -i my_song.wav -ar 16000 my_song.wav
```

### Add a start and stop song response
You need to tell your NB3 to start playing (or stop playing) the sound when it detects the appropriate voice commmand. You can do this by adding the following conditions to your code section.

```python
if best_voice_command == "start_song": # If the "best" voice command detected is "start_song"
    speaker.play_wav(sound_path)
if best_voice_command == "stop_song":  # If the "best" voice command detected is "stop_song"
    speaker.clear()
```
This code plays a WAV file (a type of sound file) by running the function: speaker.play_wav(sound_path). The file *must* be located at "sound_path", a location in your temporary sounds folder, and called "my_song.wav". This location is set at the section near the top of the code shown below:
```python
# Specify sound path(s)
# CHANGE THIS SOUND PATH TO THE NAME OF YOUR FILE HERE ------->
sound_path = f"/home/{username}/NoBlackBoxes/LastBlackBox/_tmp/sounds/my_song.wav"
```

You first need to download some sound or sound effect to this location (see the steps above). If you want to change the song or sound effect that plays, then you need to change this sound path...or add additional paths, and modify the "command response" section to play those new paths when requested.

Have Fun!
