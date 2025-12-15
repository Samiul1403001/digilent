from MyDigilent_v2 import MyDigilent
from time import sleep

FREQ = [10, 5]

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
sleep(1)

# Main send loop
try:
    for f in FREQ:
        CMD = str(f)
        msg = CMD
        Digi_1.sendStringUART(msg)
        sleep(1)

        mainloop = True
        while mainloop == True:
            RES = bytes(Digi_1.uart_read())
            if RES.decode("utf-8") == "Received":
                print("TI got the message.")
                mainloop = False
                print("")
            sleep(.5)

    Digi_1.close()
    print("Device closed.")

except KeyboardInterrupt:
    Digi_1.close()
    print("Device closed.")