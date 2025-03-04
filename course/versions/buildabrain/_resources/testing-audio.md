# Testing your NB3's Audio
The following commands will test whether your NB3's Ears (microphones) and Mouth (speaker) are working.

## Requirements
- Software installation for NB3 audio (I2S) driver. This may have been done for you.
- Hardware installation (wiring) of the NB3's ears and mouth.
- Open VS Code in the LBB repository folder
- Open a terminal window for entering commands

## Testing the Ears
The following command should be entered into the (Linux) terminal window. It will record an audio file (WAV) called "file_stereo.wav" to the current working directory of the terminal.

```bash
# Record a sound
arecord -D plughw:3 -c2 -r 48000 -f S32_LE -t wav -V stereo -v file_stereo.wav
# - Press Control+C to stop the recording
```

You should see a new file created in the VS Code "file explorer". If you select this file, then you will be able to play it back on your own computer using.

## Testing the Mouth
The following command should be entered into the (Linux) terminal window. It will playback the audio file (WAV) called "file_stereo.wav" using the NB3's speaker.

```bash
# Playback recording
aplay -D plughw:3 -c2 -r 48000 -f S32_LE -t wav -V stereo -v file_stereo.wav
# - Press Control+C to stop the playback
```

Don't worry if the sound playing is rather quiet. This is simply a "scaling" issue caused by the microphones (ears) recording a very high resolution audio file. We can adjust the volume later.

## Play some music using Python!
Use Python to play a music file.

```bash
# Play some music
python $LBB/boxes/audio/python/output/play_wav.py $LBB/_tmp/sounds/Rose_Mars_APT.wav
# - Press "q" to stop the playback
```

## Download more music (of dubious legality)
You can download music from YouTube using a handy command line tool!

Download and install the YouTube download tool.

```bash
mkdir ~/.local/bin
wget https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -O ~/.local/bin/yt-dlp
chmod a+rx ~/.local/bin/yt-dlp  # Make executable
```

Navigate to your NB3's temporary "sounds" folder using the Linux command "cd" (change directory)

```bash
cd $LBB/_tmp/sounds
```

Find a YouTube video link to your music of choice and run the following command. Be sure to replace the YouTube URL in the command below (https://www.youtube.com/watch?v=dQw4w9WgXcQ) with the link to your YouTube video.

```bash
~/.local/bin/yt-dlp -f bestaudio --extract-audio --audio-format wav --audio-quality 48K -o my_song.wav https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

Play your downloaded song using Python.

```bash
# Play some music
python $LBB/boxes/audio/python/output/play_wav.py my_song.wav
# - Press "q" to stop the playback
```
