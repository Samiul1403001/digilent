from MyDigilent import MyDigilent, clean_buffer, freq_selection_signal
from time import sleep
import numpy as np

FREQ = [1]

FREQ = np.round(FREQ, decimals=2)

# UART specs
PIN_TX = 0           # DIO pin used for UART TX (ADP3450 DIO0)
PIN_RX = 1           # optional RX pin if you want to read back
BAUDRATE = 115200
# Device object generation
Digi_1 = MyDigilent(tx=PIN_TX,
                    rx=PIN_RX,
                    baud_rate=BAUDRATE,
                    parity="none",
                    data_bits=8,
                    stop_bits=1)

Digi_1.scope_setup(channels=[1, 2])

sleep(1)

# Main loop
try:
    sample = np.zeros([61, 3])
    i = 0
    for f in FREQ:
        CMD = str(f)
        Digi_1.sendStringUART(CMD)
        sleep(0.5)
        mainloop = True
        while mainloop == True:
            RES = bytes(Digi_1.uart_read())
            if RES.decode("utf-8") == "Received":
                status = True
                sleep(0.5)
                print(f"\nMeasuring EIS at {CMD.strip()} Hz...")
                while status == True:
                    ST = bytes(Digi_1.uart_read())
                    if ST.decode("utf-8") == "Done":
                        status = False
                mainloop = False
                print(f"\nMeasuring EIS at {CMD.strip()} Hz is done.")

    Digi_1.close()
    print("Device closed.")

except KeyboardInterrupt:
    Digi_1.close()
    print("Device closed.")