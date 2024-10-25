# Timelapse on NB3 (or VK0)

## Start Timelapse
- Create the "timelapse" folder in the repo _tmp directory (_tmp/images/timelapse)
- Copy timelapse.sh script to the new folder
- Add "* * * * * /home/kampff/NoBlackBoxes/LastBlackBox/_tmp/images/timelapse/timelapse.sh 2>&1" using "crontab -e"
- ...this will take a snapshot every minute
- remove line from crontab to cancel/end timelapse acquisition

## Convert to Movie

```bash
sudo apt install ffmpeg
ffmpeg -r 30 -f image2 -pattern_type glob -i '*.jpg' -s 1920x1080 -vcodec libx264 timelapse.mp4 # 30 Hz, images, match pattern (*.jpg), HD
```

## For VK0 (long exposure)
```bash
libcamera-still --camera 1 --datetime --shutter 20000000 --gain 8 --awbgains 2,1.81 --immediate --denoise cdn_hq
```
