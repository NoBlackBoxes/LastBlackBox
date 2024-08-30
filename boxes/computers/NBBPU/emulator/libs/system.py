# System Library
import numpy as np

# Define system state
class State:
    pc = 0
    registers = np.zeros(16, dtype=np.int16)
    ram = np.zeros(256, dtype=np.int16)
state = State()
