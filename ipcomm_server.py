from MyDigilent import MyDigilent, FFT, clean_buffer, freq_selection_signal, dual_phase_demod
from time import sleep
import numpy as np, socket, struct

# --- TCP CONFIGURATION ---
TCP_IP = '127.0.0.1'  # Localhost (use Host PC IP if running in ADP3450 Linux Mode)
TCP_PORT = 5005       # Arbitrary port (must match receiver)

print(f"Connecting to server at {TCP_IP}:{TCP_PORT}...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    sock.connect((TCP_IP, TCP_PORT))
    print("Connected!")
except ConnectionRefusedError:
    print("Failed to connect. Is the Receiver script running?")
    quit()