import time
from WF_SDK import device
from WF_SDK.protocol import uart

# ------------------- USER SETTINGS -------------------
PIN_TX = 0           # DIO pin used for UART TX (ADP3450 DIO0)
PIN_RX = 1           # optional RX pin if you want to read back
BAUDRATE = 115200
SEND_VALUE = "Hello"
SEND_INTERVAL = 10.0  # seconds
# ------------------------------------------------------

print("Opening ADP3450 device...")
dev = device.open()      # open first connected device

# Configure UART on the chosen DIO pins
uart.open(
    device_data=dev,
    tx=PIN_TX,
    rx=PIN_RX,
    baud_rate=BAUDRATE,
    parity="none",
    data_bits=8,
    stop_bits=1
)
print(f"UART initialized on DIO{PIN_TX} (TX) @ {BAUDRATE} baud")

# Main send loop
try:
    while True:
        message = str(SEND_VALUE) + "\n"
        uart.write(dev, message.encode())
        print(f"Sent: {message.strip()}")
        time.sleep(SEND_INTERVAL)
except KeyboardInterrupt:
    print("\nStopped by user.")
    uart.close(dev)
    device.close(dev)
    print("Device closed.")