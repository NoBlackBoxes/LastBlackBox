import spidev
import time

# MAX7219 Register Addresses
REG_NO_OP = 0x00
REG_DIGIT0 = 0x01
REG_DIGIT1 = 0x02
REG_DIGIT2 = 0x03
REG_DIGIT3 = 0x04
REG_DIGIT4 = 0x05
REG_DIGIT5 = 0x06
REG_DIGIT6 = 0x07
REG_DIGIT7 = 0x08
REG_DECODE_MODE = 0x09
REG_INTENSITY = 0x0A
REG_SCAN_LIMIT = 0x0B
REG_SHUTDOWN = 0x0C
REG_DISP_TEST = 0x0F

class MAX7219:
    def __init__(self, bus=0, device=0):
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = 1000000
        self.spi.mode = 0b00
        self.initialize()

    def send(self, register, data):
        """Send a command to the MAX7219."""
        print(f"Sending: Register {register:#04x}, Data {data:#04x}")
        self.spi.xfer2([register, data])

    def initialize(self):
        """Initialize the MAX7219."""
        print("Initializing MAX7219...")
        self.send(REG_SHUTDOWN, 0x01)  # Exit shutdown mode
        self.send(REG_DISP_TEST, 0x00)  # Disable display test
        self.send(REG_SCAN_LIMIT, 0x07)  # Scan all digits
        self.send(REG_DECODE_MODE, 0x00)  # No decode mode
        self.send(REG_INTENSITY, 0x08)  # Medium intensity
        self.clear()

    def clear(self):
        """Clear all LEDs."""
        print("Clearing display...")
        for i in range(1, 9):
            self.send(i, 0x00)

    def set_all(self, state):
        """Turn all LEDs on or off."""
        data = 0xFF if state else 0x00
        print(f"Setting all LEDs {'ON' if state else 'OFF'}")
        for i in range(1, 9):
            self.send(i, data)

    def close(self):
        """Close the SPI connection."""
        print("Closing SPI connection.")
        self.spi.close()

# Main Program
if __name__ == "__main__":
    try:
        max7219 = MAX7219(device=0)

        while True:
            print("Turning all LEDs ON")
            max7219.set_all(True)
            time.sleep(2)

            print("Turning all LEDs OFF")
            max7219.set_all(False)
            time.sleep(2)

    except KeyboardInterrupt:
        print("Exiting program.")
    finally:
        max7219.clear()
        max7219.close()
