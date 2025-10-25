import time
from WF_SDK import device
from WF_SDK.protocol import uart

def sendStringUART(dev, section):
    i = 0
    while i < 8:
        if i < len(section):
            uart.write(dev, section[i])
        else:
            uart.write(dev, "\0")
        i += 1
    # uart.write(dev, section[i])

# ------------------- USER SETTINGS -------------------
PIN_TX = 0           # DIO pin used for UART TX (ADP3450 DIO0)
PIN_RX = 1           # optional RX pin if you want to read back
BAUDRATE = 115200
FREQ = [100, 200, 300, 400, 500]
CMD = ""
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
    while CMD != "end":
        CMD = input("\nEnter desired frequency: ")
        msg = CMD
        sendStringUART(dev, msg)
        time.sleep(1)
        while True:
            RES = bytes(uart.read(dev))
            if str(RES) == "Received":
                print(f"\nMeasuring EIS at {CMD.strip()} Hz...")
                time.sleep(3)
            elif str(RES) == "Done":
                break
        print(f"\n\nDone measuring EIS at {CMD.strip()} Hz! Going for the next one...\n")
        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopped by user.")
    uart.close(dev)
    device.close(dev)
    print("Device closed.")