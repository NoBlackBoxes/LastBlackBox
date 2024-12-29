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
ser.baudrate = 115200
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
            screen.addstr(0, 0, 'STOP ')
            ser.write(b'x')
            time.sleep(0.05)
        elif char == curses.KEY_RIGHT:
            screen.addstr(0, 0, 'right')
            ser.write(b'r')
            time.sleep(0.05)
        elif char == curses.KEY_LEFT:
            screen.addstr(0, 0, 'left ')       
            ser.write(b'l')
            time.sleep(0.05)
        elif char == curses.KEY_UP:
            screen.addstr(0, 0, 'up   ')       
            ser.write(b'f')
            time.sleep(0.05)
        elif char == curses.KEY_DOWN:
            screen.addstr(0, 0, 'down ')
            ser.write(b'b')
            time.sleep(0.05)
finally:
    # shut down
    curses.nocbreak(); screen.keypad(0); curses.echo()
    curses.endwin()
    ser.close()

#FIN

