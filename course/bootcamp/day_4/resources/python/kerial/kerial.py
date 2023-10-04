import serial
import time
import curses

# get the curses screen window
screen = curses.initscr()

# turn off input echoing
curses.noecho()

# respond to keys immediately (don't wait for enter)
curses.cbreak()

# map arrow keys to special values
screen.keypad(True)

# Configure serial port
ser = serial.Serial()
ser.baudrate = 19200
ser.port = '/dev/ttyUSB0'

# Open serial port
ser.open()
time.sleep(2.00) # Wait for connection before sending any data

try:
    while True:
        char = screen.getch()
        if char == ord('q'):
            break
        elif char == ord('x'):
            screen.addstr(0, 0, 'OFF')
            ser.write(b'x')
            time.sleep(0.05)
        elif char == ord('o'):
            screen.addstr(0, 0, 'ON')
            ser.write(b'o')
            time.sleep(0.05)
finally:
    # shut down
    curses.nocbreak(); screen.keypad(0); curses.echo()
    curses.endwin()
    ser.close()

#FIN

