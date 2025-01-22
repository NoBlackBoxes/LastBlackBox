import spidev
import time
import numpy as np

# MAX7219 Register Addresses
REG_NO_OP       = 0x00
REG_DIGIT0      = 0x01
REG_DIGIT1      = 0x02
REG_DIGIT2      = 0x03
REG_DIGIT3      = 0x04
REG_DIGIT4      = 0x05
REG_DIGIT5      = 0x06
REG_DIGIT6      = 0x07
REG_DIGIT7      = 0x08
REG_DECODE_MODE = 0x09
REG_INTENSITY   = 0x0A
REG_SCAN_LIMIT  = 0x0B
REG_SHUTDOWN    = 0x0C
REG_DISP_TEST   = 0x0F

#
# Eye (8x8 Led Matrix with MAX2719 driver) Class
#
class Eye:
    def __init__(self, device):
        self.spi = spidev.SpiDev()
        self.spi.open(0, device)
        self.spi.max_speed_hz = 1000000 #   1 MHz (surely it can go faster)
        self.spi.mode = 0b00                # SPI mode 0 (CPOL=0, CPHA=0)
        self.buffer = np.zeros((8,8), dtype=bool)
        self.pupil_x = 3.0
        self.pupil_y = 3.0
        self.initialize()

    def initialize(self):
        """Initialize the MAX7219."""
        #print("Initializing MAX7219...")
        self.send(REG_SHUTDOWN, 0x01)       # Exit shutdown mode
        self.send(REG_DISP_TEST, 0x00)      # Disable display test
        self.send(REG_SCAN_LIMIT, 0x07)     # Scan all digits
        self.send(REG_DECODE_MODE, 0x00)    # No decode mode
        self.send(REG_INTENSITY, 0x08)      # Medium intensity
        self.clear()

    def send(self, register, data):
        """Send a command to the MAX7219."""
        #print(f"Sending: Register {register:#04x}, Data {data:#04x}")
        self.spi.xfer2([register, data])

    def clear(self):
        """Clear all LEDs."""
        #print("Clearing display...")
        for i in range(1, 9):
            self.send(i, 0x00)

    def set_all(self, state):
        """Turn all LEDs on or off."""
        data = 0xFF if state else 0x00
        #print(f"Setting all LEDs {'ON' if state else 'OFF'}")
        for i in range(1, 9):
            self.send(i, data)

    def set_intensity(self, intensity):
        """
        Set the LED intensity (brightness) for the MAX7219.

        :param intensity: Brightness level (0x00 to 0x0F).
        :raises ValueError: If the intensity is out of the valid range.
        """
        if not (0x00 <= intensity <= 0x0F):
            raise ValueError("Intensity must be between 0x00 (min) and 0x0F (max).")
        #print(f"Setting intensity to {intensity:#02x}")
        self.send(REG_INTENSITY, intensity)

    def set_pixel(self, x, y, state):
        """
        Set the state of a single LED in the matrix.

        :param x: X-coordinate (0-7).
        :param y: Y-coordinate (0-7).
        :param state: True for ON, False for OFF.
        """
        if 0 <= x < 8 and 0 <= y < 8:
            self.matrix[x, y] = state

    def set_row(self, row, state):
        """
        Set the state of all LEDs in a row.

        :param row: row index (0-7).
        :param state: True for ON, False for OFF.
        """
        if 0 <= row < 8:
            self.matrix[:, row] = state

    def set_col(self, col, state):
        """
        Set the state of all LEDs in a column.

        :param col: col index (0-7).
        :param state: True for ON, False for OFF.
        """
        if 0 <= col < 8:
            self.matrix[col, :] = state

    def refresh(self):
        """
        Send the current state of the matrix to the MAX7219.
        """
        for row in range(8):
            # Convert the boolean row to a byte (each bit represents an LED)
            data = int("".join(str(int(b)) for b in self.matrix[row]), 2)
            self.send(REG_DIGIT0 + row, data)
    
    def gaze(self, x, y):
        self.matrix = np.zeros((8,8), dtype=bool) # Clear matrix
        # Draw socket
        self.set_row(0, True)
        self.set_row(7, True)
        self.set_col(0, True)
        self.set_col(7, True)
        self.set_pixel(0,0,False)
        self.set_pixel(7,0,False)
        self.set_pixel(0,7,False)
        self.set_pixel(7,7,False)
        # Draw pupil
        if x-1 > 0:
            self.set_pixel(x-1, y, True)
            if y-1 > 0:
                self.set_pixel(x, y-1, True)
                self.set_pixel(x-1, y-1, True)
        self.set_pixel(x, y, True)
        self.refresh()

    def saccade(self, x, y, duration):
        # in 10 ms steps, move gaze to new location
        steps = duration // 10
        dx = float(x - self.pupil_x) / steps
        dy = float(y - self.pupil_y) / steps
        for i in range(steps+1):
            time.sleep(0.010)
            self.pupil_x = self.pupil_x + dx
            self.pupil_y = self.pupil_y + dy
            self.gaze(int(self.pupil_x),int(self.pupil_y))
            print(i,dx, self.pupil_x, self.pupil_y)

    def close(self):
        """Close the SPI connection."""
        #print("Closing SPI connection.")
        self.spi.close()

# FIN