# The Last Black Box: Course: Scripts: Videos
Generate the LBB video list (with links) and thumbnail frames

## Normalize Audio
Use FFMPEG's loudnorm filter (via a Python wrapper: https://github.com/slhck/ffmpeg-normalize/tree/master)

```bash
# Requires an installation of FFMPEG
pip install ffmpeg-normalize

# Usage
ffmpeg-normalize -p -v -f -c:a aac -b:a 320k -t -17 --keep-loudness-range-target <filename.mkv>

# Detect volume levels
ffmpeg -af "volumedetect" -vn -sn -dn -f null - -i <filename.mkv>

## Mean output volume should be around -20dB (to be confirmed)
```

