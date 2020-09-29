# Deep learning on your Pi
So far you've programmed in some interesting behaviours, and mayeb your bot has some semblances of intelligence.  Now its time to step that up with the [Coral USB accelerator board](https://coral.ai/docs/accelerator/get-started "DeEp LeArNiNg").  Unfortunately, setting the board up requires access to the web - you might find it beneficial to plug your robot Pi into your monitor so you don't have to remote in for this part.  Some of you may be tempted to install the maximum frequency runtime for the board.  Just be careful you're not running the camera cable too close to the board, and accept that you'll dramatically reduce your battery time.

## Deploy a pre-baked model on your accelerator

Feel free to go through any of the examples on their [website](https://coral.ai/examples/#project-tutorials "examples").  I've tried the BodyPix example and its pretty cool.

Begin by cloning the repository into a directory of your choice

`git clone https://github.com/google-coral/project-bodypix.git`

You should now be able to follow the instructions on the github page and run the algorithm and check out your style.  Use this as an opportunity to debug any issues with your edge TPU installation.  Some issues I found and seem to be common
1.  `no module named 'edgetpu'`
- This can be solved by running `sudo apt-get install python3-edgetpu`

2. `Error in device opening (/sys/bus/usb/devices/2-2)!`
- This can be solved by running the program with `sudo`, but this can cause module import errors since your search path will change, annoyingly.
-

3. `ModuleNotFoundError`
- As opposed to the edgetpu error where a linux library needs to be installed, most of these can be resolved by using `pip3 install <module>`.  Make sure you use `pip3` and not `pip`, otherwise you will only install for python2(.7).  If it still reports that the modules are missing, check the path that Python3 is using to search for the modules.  Talk to us if you're unsure.

### Visualising the results

If you're still working natively on your robot pi with the monitor plugged in, you should be able to see the segmentation live.  If you're remoting in, you won't.  At this point, all parties will be wanting to figure out a way to view the video stream when the monitor isn't plugged in.  There are 3 options for this...
1. X11 Forwarding.  This is a common protocol for displaying graphical interfaces on a remote machine.
2. Broadcast the output of the video to a hosted website, like we did earlier
3. Something else.  Perhaps forwarding the raw data of a TCP connection using the server/client we set up earlier.

