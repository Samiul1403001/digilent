#!/usr/bin/env python3
# save as adp_uart_test_send.py
import time
from WF_SDK import device
from WF_SDK.protocol import uart

PIN_TX = 0
PIN_RX = 1
BAUDRATE = 115200

dev = device.open()

uart.open(device_data=dev, tx=PIN_TX, rx=PIN_RX, baud_rate=BAUDRATE,
          parity="none", data_bits=8, stop_bits=1)

try:
    while True:
        uart.write(dev, b"A")   # send single byte 'A'
        print("sent 'A'")
        time.sleep(1.0)
except KeyboardInterrupt:
    print("Disconnecting device...")
    uart.close(dev)
    device.close(dev)
