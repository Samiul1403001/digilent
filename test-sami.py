from time import sleep
from WF_SDK import device, scope
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
        sleep(1)
        while True:
            RES = bytes(uart.read(dev))
            if RES.decode("utf-8") == "Received":
                mainloop = True
                time = []
                print(f"\nMeasuring EIS at {CMD.strip()} Hz...")
                while RES.decode("utf-8") != "Done":
                    # initialize the scope with default settings
                    # choose sensible values
                    samp_freq = 1e6       # 1 MHz sampling
                    buf_size = 5
                    scope.open(dev, sampling_frequency=samp_freq, buffer_size=buf_size, offset=0, amplitude_range=5)
                    sleep(1)

                    current = scope.record(dev, channel=1)
                    volt_1 = scope.record(dev, channel=2)

                    # generate buffer for time moments
                    # for index in range(len(current)):
                    #     time.append(index * 1e03 / scope.data.sampling_frequency)
                    print("current value: ", current, "A")
                    sleep(1)
                    RES = bytes(uart.read(dev))
            if mainloop == True:
                scope.close(dev)
                break
        sleep(1)
        print(f"\n\nDone measuring EIS at {CMD.strip()} Hz! Going for the next one...\n")
        sleep(1)

except KeyboardInterrupt:
    print("\nStopped by user.")
    uart.close(dev)
    scope.close(dev)
    device.close(dev)
    print("Device closed.")