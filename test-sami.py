#!/usr/bin/env python3
"""
Send the number 150 every 10 seconds over UART using Digilent ADP3450 DIO pins.
Works entirely inside the ADP3450 (Linux mode) using WF_SDK.

Hardware wiring:
  ADP DIO_TX (e.g. DIO0) -> MCU RX
  ADP GND                -> MCU GND

Note: set your digital voltage (Vdig) to match MCU logic (usually 3.3 V).
"""

import time
from WF_SDK import device, supplies
from WF_SDK.protocol import uart

# ------------------- USER SETTINGS -------------------
PIN_TX = 0           # DIO pin used for UART TX (ADP3450 DIO0)
PIN_RX = 1           # optional RX pin if you want to read back
BAUDRATE = 115200
SEND_VALUE = 150
SEND_INTERVAL = 10.0  # seconds
# ------------------------------------------------------

print("Opening ADP3450 device...")
dev = device.open()      # open first connected device

# # (optional) ensure digital voltage rail is 3.3 V
# supplies.switch(dev, True)          # enable supplies
# supplies.set_mode(dev, "Digital", "fixed")
# supplies.set_voltage(dev, "Digital", 3.3)
# print("Set Vdig to 3.3 V")

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
finally:
    uart.close(dev)
    device.close(dev)
    print("Device closed.")
