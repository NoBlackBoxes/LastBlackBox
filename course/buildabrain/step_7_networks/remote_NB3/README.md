# The Last Black Box: *Build a Brain*

## Project: Remote NB3

Let's build a remote-controlled WiFi connected robot with a camera. You can drive it (and see what it sees) anywhere you have an internet connection!

## Step 1 - Power

We will need more power to run the bigger computer, WiFi, and camera. Let's add a power supply.

<p align="center">
<img src="resources/images/NB3_power_wiring.png" alt="NB3 power wiring" width="400" height="300">
</p>

- Watch this video: [NB3 Power](https://vimeo.com/626839902)
- ***Task***: Add a (regulated) 5 volt power supply to your robot, which you can use while debugging to save your AA batteries and to provide enough power for the Raspberry Pi computer.
- *NOTE*: Your NB3_power board cable *might* have inverted colors (black to +5V, red to 0V) relative to that shown in the assembly video. This doesn't matter, as the plugs will only work in one orientation and the correct voltage is conveyed to the correct position on the body.

## Step 2: Computer

- Watch this video: [NB3 midbrain](https://vimeo.com/627777644)

- ***Task***: Mount a Raspberry Pi on your robot (and connect its power inputs using your *shortest* jumper cables, 2x5V and 2x0V from the NB3, to the correct GPIO pins on the RPi...please *double-check* the pin numbers)
  - This pinout of the Raspberry Pi GPIO might be useful: [Raspberry Pi GPIO](resources/images/rpi_GPIO_pinout.png)
- ***Task 2***: Copy a version of the Raspberry Pi OS (operating system) to your micro-SD card
  - We currently recommend using the [Raspberry Pi Imager Tool](https://www.raspberrypi.com/software/)
    - Please be **sure** to do the following before burning your image:
      1. Choose the 64-bit version of Raspberry Pi OS
      2. Set a hostname (for your NB3)
      3. Set a username (for you) and password (DO NOT FORGET IT!)
      4. Enable SSH
      5. Set the WiFi network name and password
      6. Set your country/locale

## Step 3: Connecting

We will be using VS code as our tool for writing programs. VS code is just a simple, but really powerful, text editor. It has extensions for all sorts of useful tools (Python, Git, etc.), and most importantly for today's tasks, connecting to a "remote" computer (our NB3's Raspberry Pi).

  - Download Visual Studio Code to your main (laptop/desktop) computer: [VS Code](https://code.visualstudio.com/Download)

- ***Task 3***: Connect to your Raspberry Pi from your main computer.
  - Turn on your Raspberry Pi and *hope* that it automatically connects to your WiFi network.
  - ***You will still need to find out the IP address of your Raspberry Pi*** before you can connect to it via SSH. How do you find this out?
    - *Make sure your main computer (laptop/desktop) is connected to the ***same*** network as your RPi.* (i.e. the same WiFi network)
    - If you have access to your WiFi router, then you can check for any *new* devices that connect when you turn on your RPi...that will reveal the IP address that your RPi was "assigned" when it connected.
    - If you have a micro-HDMI cable and a spare monitor/TV, then you can connect it before the RPi boots and watch the "scrolling text logs" of the Linux OS while it boots up. At the very end, there will be a line that says..."connected, etc. IP addres: XXX.XXX.XXX.XX". Then you know the IP address.
    - There are *many* other ways. Let us know what works for you!
    - When you know your IP address, use VS Code's "Remote - SSH" extension to connect.
      1. Install the "Remote - SSH" extension (search for it in the left-hand toolbar "Extensions" menu)
      2. Click on the little green box in the lower left corner and select "Connect to Host".

        <p align="center">
        <img src="resources/images/remote-ssh-button.png" alt="Remote SH" width="220" height="125">
        </p>

      3. Enter the following in the text box that appears:

      ```bash
      ssh your-username@your-IP-address
      # Example: ssh adam@192.168.1.121

      # It may also work, without knowing your IP, to enter the following
      ssh your-username@your-NB3-hostname
      # Example: ssh adam@myNB3
      ```
    - If all goes well, then you should be able to open a "Terminal" window in VS code that is *actually* a command line terminal running on your NB3. You can then continue with today's tasks.
    - If all *does not* go well, and it often doesn't, then give us a shout!

## Step 4: Operating Systems

Now that we have a connection to our NB3 Linux computer (Raspberry Pi), we can *finally* start learning about Linux.

Linux is an open-source Operating System, which means you can look at ***all*** of the code that makes it work.

We will use Linux through a "command line terminal". This means we will be typing text commands to do all of things that you usually do with a mouse or touchscreen (write text files, make folders, save pictures, etc.). This can feel weird, but it can also be very efficient...if you spend a little bit of time to get familiar with the commands.

Open a terminal on your NB3 (inside VS code) and let's test out some of the basic Linux commands.

```bash
# At the command prompt, type the following commands...

mkdir test_folder
# You just made a folder (directory)

ls
# This lists the contents (files and folders) in the current directory, you should see the folder you just made

pwd
# This prints the working directory (the folder you are currently in). It should print something like /home/<your username>/

cd test_folder
# Change the working directory to "test_folder"

pwd
# You should now see that the working directory has changed to test_folder

cd ..
# This goes "up" one directory, i.e. steps out of the "test_folder"

pwd
# You should be back "home"

# Let's do something FUN!
htop
# This will run a program that shows you all of "processes" running on your NB3. At the top, 
#  it will show you how much each CPU core is being used. There is a lot of other information. 
#  It can be very useful to see what is going on (and what is slowing down) your NB3

# press 'q' or F10 to quit htop
```

Let's create a simple python progam using the Linux command line terminal.


```bash
nano test.py
# This creates a file called test.py and opens it in the nano text editor
```

You will see a screen like this now:

<p align="center">
<img src="resources/images/nano_screenshot.png" alt="Nano Text Editor" width="600" height="300">
</p>

You can type text as usual into this window. Let's add some simplePython code. Copy the following text into you Nano text editor.

```python
for i in range(5):
  print("Hello!")
print("from NB3")
```

You can close and save the file by pressing Ctrl+x. It will ask if you want to save, press 'Y' for yes and confirm the name by pressing 'Enter'.

You will now be back in the command line terminal.

```bash
# At the command prompt, type the following commands...

ls
# This will list the folder contents, and you should see you test.py file.

python test.py
# This will run the code in test.py using the Python interpreter
```

There is obviously quite a bit more you can do with Linux. Google and ChatGPT are your friends...just make sure you ask for "command line examples".

## Step 5: Git

Git is a piece of software that keeps track of any changes you make to the files within a folder (repository) filled with code. This is used to manage different versions of the code and to allow many different people to contribute their own changes in an orderly manner. GitHub, the website we have been using throughout this course, is an online service for "hosting" Git repositories.

Let's get a copy of the *entire* course repository onto our NB3. This is called "cloning" the repository.

```bash
# Install git
sudo apt-get install git # This command installs the Git program on Linux. 
# Note 1: "apt" is a package manager and will be used a lot to download software for Linux
# Note 2: In order to install software, you must have "super user" permissions. 
#   "sudo" means "super user do" and lets your user run commands as a super user, 
#   which is why you will need to enter your password.

# Create a new folder in your Home directory and change into it
cd ~                # Go to your home directory (this a quick way to get back home from any working directory)
mkdir NoBlackBoxes  # Create a NoBlackBoxes directory
cd NoBlackBoxes     # Change to the NoBlackBoxes directory

# Clone LBB repo
git clone --depth 1 https://github.com/NoBlackBoxes/LastBlackBox
```

You will now have a copy of the entire LastBlackBox course on your NB3.

## Step 6: Coding (Arduino)

We will soon see what we can do with Python!

...but first, to make things interesting, you should upload the following code to your ***Arduino*** using your laptop/desktop and the Arduino IDE.

```c
/*
  Serial Server
     - Respond to single character commands received via serial
*/

void setup() {
  // Initialize serial port
  Serial.begin(19200);

  // Initialize output pins
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  
  // Check for any incoming bytes
  if (Serial.available() > 0) {
    char newChar = Serial.read();

    // Respond to command "x"
    if(newChar == 'x') {
      // Turn off LED pin 13
      digitalWrite(LED_BUILTIN, LOW);
    }

    // Respond to command "o"
    if(newChar == 'o') {
      // Turn on LED pin 13
      digitalWrite(LED_BUILTIN, HIGH);
    }

  }

  // Wait a bit
  delay(10);
}
```

This code tells your Arduino to listen to the serial port. If it recives the letter 'o', turn on the built-in LED. If it receives the letter 'x', turn it off.

## Step 7: Coding (Python)

We will write some simple Python code to talk to our Arduino (hindbrain) from our Raspberry Pi (midbrain) via the serial port. This means ***we must connect our Raspberry Pi to our Arduino with a USB cable***. You can simply unplug the USB cable from your laptop and plug it into the Raspberry Pi.

To talk "serial" in Python, we will need to install an extra library.

```bash
# Do ONE of the following

sudo apt-get install python3-serial
# This uses Linux's apt package manager to install the pyserial library

# Install Pip
sudo apt-get install python3-pip
pip install pyserial
# This uses Python's pip package manager to install the pyserial library
```

Now let's look at some code in the LastBlackBox repository.

```bash
cd ~  # Make sure you are in your home directory
cd NoBlackBoxes/LastBlackBox/course/bootcamp/day_4/resources/python/serial # Change to a folder with some Python code
ls # List the contents of this folder. You will see a file called serial_write.py
less serial_write.py # This will print the contents of the python file to your terminal screen
```

This is what the file looks like...

```python
import serial
import time

# Configure serial port
ser = serial.Serial()
ser.baudrate = 19200
ser.port = '/dev/ttyUSB0'

# Open serial port
ser.open()
time.sleep(2.00) # Wait for connection before sending any data

# Send a character
ser.write(b'x')
time.sleep(0.05)

# Send a character
ser.write(b'o')
time.sleep(1.00)

# Send a character
ser.write(b'x')
time.sleep(0.05)

# Close serial port
ser.close()
```

The above Python file imports the time and serial libraries. Opens a serial connection (to your Arduino). Sends the letter 'x', waits for 50 ms, sends the letter 'o', waits for 1 second, then sends the letter 'x' again. It then closes the serial connection.

```bash
# Run the Python code like this
python serial_write.py
```
If all goes well, then you should see the LED on your Arduino turn on for 1 second when you run the code.

Try changing the code to make the LED stay on longer...

If you ***understand*** this code (and the code running on your Arduino), then you are ready to build a remote-controlled NB3!

## Step 8: Websites

This is a website. Just a text file written in the hyper-text markup language (HTML) that your "web browser" knows how to display in a nicely formatted way.

```html
<!DOCTYPE html>
<html>

<body>
This is a <b>website</b>
</body>

</html>
```

- ***Task***: Make your own website (an HTML file). Open it in your web browser.

## Step 9: Servers

A server is just a computer that you can connect to and ask it to send (serve) you a file (e.g. an HTML file).

Your "web browser" is a program for asking for HTML files from servers (using the the hyper-text transfer protocol, HTTP) and then displaying them in a pretty manner.

Let's write our *own* HTTP server and run it on our Raspberry Pi (NB3)!

- ***Task 1***: Run your own server (Run this python code on your NB3, put the [index.html](resources/index.html) in the same folder as your python file).

```python
# Simple HTTP server
import os
import socket

# Load your HTML page (index.html)
path = 'index.html'
file = open(path,"r")
html = file.read()
file.close()

# Set host address (your NB3's IP) and an unused port (1234)
HOST, PORT = '', 1234

# Open a "listening" socket (waits for external connections)
listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listen_socket.bind((HOST, PORT))
listen_socket.listen(1)
print(f'Serving HTTP on port {PORT} ...')

# Serve incoming connections
while True:
    # Listen for a connection
    client_connection, client_address = listen_socket.accept()

    # When a connection arrives, retrieve/decode HTTP request
    request_data = client_connection.recv(1024)
    request_text = request_data.decode('utf-8')
    print(request_text)

    # (Optional) parse request
    #   - if you want to serve different files based on the request

    # Respond to target (send the bytes of your HTML file)
    http_response = bytes(html, 'utf-8')
    client_connection.sendall(http_response)

    # Close client connection
    client_connection.close()

#FIN
```

- ***Task 2***: Use the web browser on your laptop/desktop to "navigate" to the URL (universal record locator) at your NB3's IP address and the PORT specified in the Python program (1234):

```bash
http://IP-ADDRESS:1234

# Example: http://192.168.1.157:1234 
```

***This is the internet.***

## Step 10: Cameras

Let's add a camera (eyes) to your NB3!

- Watch this video: [NB3 Forebrain](https://vimeo.com/628545232)

----

When your camera is mounted and connected, you can test whether it works by running the following command from the Linux terminal.

```bash
libcamera-still -o test.png
```

This command will take a picture and save it as a PNG in the same folder where you run the command. You can open it directly in VS Code.

## Step 11: Computer Vision

Taking a picture is just the first step in making a computer "see". How do we extract useful information from the image? How do we detect movement? How do we identify and localise different objects? Many of these tasks are surprisingly hard and remain unsolved problems in both engineering and neuroscience?

Let's get started.

The first step will be acquiring an image using Python (so we can then manipulate and analyse the image using code).

### Acquire an image using Python

- Run this [Example](resources/python/camera/capture_save.py) code to acquire a still image and save it to a file.

### Process an image using Python and the Numpy library

- Run this [Example](resources/python/camera/capture_process_save.py) code to acquire a still image and save it to a file.
- Note: you may need to install another library to save the numpy array as an image

  ```bash
  pip install pillow
  ```

### Stream images from your NB3 camera to the local network (so we can view the live stream on your laptop)

- Run this [Example](resources/python/camera/capture_stream.py) code to continously acquire images and stream them to a website.

## PROJECT

Build a remote-controlled NB3 (that streams live video of what is sees!) 

- This example of using the keyboard to send commands via SSH will be very helpful: [Drive](resources/python/drive/drive.py)

    ```bash
    # You must install the SSH keyboard Python library
    pip install sshkeyboard
    ```

- This code should be uploaded to your Arduino for the above Python code to work: [Serial Controller](resources/arduino/serial_controller/serial_controller.ino)