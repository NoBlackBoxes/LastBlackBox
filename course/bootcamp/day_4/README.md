# The Last Black Box : *Bootcamp* : Day 4 - Signal Processing

## Morning

----

### Networks

***Everyone must have their NB3 connected to the WiFi network and a "Remote - SSH" connection from VS Code***

- Lecture: "Communication, protocols, and the internet"

### Linux

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

### Git

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

# Clone LBB repo with all of its submodules
git clone --depth 1 https://github.com/NoBlackBoxes/LastBlackBox
```

You will now have a copy of the entire LastBlackBox course on your NB3.

### Python (but first some more Arduino)

We will soon see what we can do with Python!

...but first, to make things interesting, you should upload the following code to your ***Arduino*** using your laptop and the Arduino IDE.

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

### Python

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

### Morning Project: *Suggested steps*

- ***Task 1***: Design and implement a protocol for communicating between your midbrain and hindbrain
- ***Project***: Build a remote control robot


  1. Tell your Arduino to listen for "serial commands" coming from either your laptop or RPi - [Example](resources/arduino/serial_server)
  2. Send serial commands from you laptop/RPi using python (pyserial) - [Example](resources/python/serial)
  3. Use python to detect keypresses... - [Example](resources/python/keyboard)
  4. ...and send corresponding serial commands in response - [Example](resources/python/kerial)
  5. Extend the Arduino code to respond to specific serial commands (usually single letters) with specific movements - [Example](resources/arduino/serial_controller)
  6. Find a way to use keypresses to send the correct serial commands to Arduino so you can drive it around at will! - [Example](resources/python/drive)

----

## Afternoon

----


### Websites

This is a website. Just a text file written in the hyper-text markup language (HTML) that your "web browser" knows how to display in a nicely formatted way.

```html
<!DOCTYPE html>
<html>

<body>
This is a <b>website</b>
</body>

</html>
```

- ***Task 1***: Make your own website (an HTML file). Open it in your web browser.

### Servers

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

- ***Task 2***: Use the web browser on your laptop to "navigate" to the URL (universal record locator) at your NB3's IP address and the PORT specified in the Python program (1234):

```bash
http://IP-ADDRESS:1234

# Example: http://192.168.1.157:1234 
```

***This is the internet.***

----

## Evening

----
