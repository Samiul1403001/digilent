from ctypes import (c_int, c_byte, c_ubyte,
    cdll, byref, c_double, create_string_buffer, c_bool)
import math
import sys
import time
from dwfconstants import (filterDecimate,
    AnalogOutNodeCarrier,funcDC,DwfStateDone)
import random

if sys.platform.startswith("win"):
    dwf = cdll.LoadLibrary("dwf.dll")
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

hdwf = c_int()

print("Opening first device")
dwf.FDwfDeviceConfigOpen(c_int(-1), c_int(0), byref(hdwf)) 

if hdwf.value == 0:
    print("failed to open device")
    szerr = create_string_buffer(512)
    dwf.FDwfGetLastErrorMsg(szerr)
    print(str(szerr.value))
    quit()

print("Configuring the instrument")

print("Configuring SPI...")
# set the SPI frequency to 1000000 Hz
dwf.FDwfDigitalSpiFrequencySet(hdwf, c_double(1e6))

# set DIO channel 1 as the SPI clock
dwf.FDwfDigitalSpiClockSet(hdwf, c_int(1))

# set DIO channel 2 as the SPI MOSI pin
dwf.FDwfDigitalSpiDataSet(hdwf, c_int(0), c_int(2))

# set DIO channel 3 as the SPI MISO pin 
dwf.FDwfDigitalSpiDataSet(hdwf, c_int(1), c_int(3))

# set the SPI mode to 3, where CPOL = 1 and CPHA = 1
dwf.FDwfDigitalSpiModeSet(hdwf, c_int(3))

# set the bit order for SPI data to MSB
dwf.FDwfDigitalSpiOrderSet(hdwf, c_int(1))

# set the value of DIO channel 0, connected as SPI chip select, to high 
dwf.FDwfDigitalSpiSelect(hdwf, c_int(0), c_int(1))

# cDQ 0 SISO, 1 MOSI/MISO, 2 dual, 4 quad
#                                cDQ       bits 0    data 0
dwf.FDwfDigitalSpiWriteOne(hdwf, c_int(0), c_int(0), c_int(0)) # start driving the channels
time.sleep(1)

print("Configuring Analog In...")

# set the sampling frequency to 20 MHz
dwf.FDwfAnalogInFrequencySet(hdwf, c_double(20000000.0))

# set buffer size to 4000 samples
dwf.FDwfAnalogInBufferSizeSet(hdwf, c_int(4000))

# enable Analog channel 1
dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(-1), c_bool(True))

# set channel 1 input range to 5 volts
dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(-1), c_double(5))

# enable decimation on analog in channel 1
dwf.FDwfAnalogInChannelFilterSet(hdwf, c_int(-1), filterDecimate)
time.sleep(2) # wait for the offset to stabalize. 

print("Configuring Power Supply....")

# enable positive supply
dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(0), c_int(0), c_double(True))

# set voltage to 3.3V
dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(0), c_int(1), c_double(3.3))

# master enable
dwf.FDwfAnalogIOEnableSet(hdwf, c_int(True))
time.sleep(5)

print("Configuring the ADC")
print("Writing config...")

# write to the config register, setting the gain to 1, the active channel to 0
b = (c_ubyte*4)(0x10,0x00,0x01,0x10)

# set DIO channel 0, operating as SPI CS to low (0)
dwf.FDwfDigitalSpiSelect(hdwf, c_int(0), c_int(0))

# perform a SPI write operation over DQ line 0, with 8 bits per word, sending the above buffer (b)
dwf.FDwfDigitalSpiWrite(hdwf, c_int(0), c_int(8), b, c_int(len(b)))

# set DIO channel 0, operating as SPI CS to high (1)
dwf.FDwfDigitalSpiSelect(hdwf, c_int(0), c_int(1))

time.sleep(.05)

# write to the mode register, setting averaging to 1
b = (c_byte*4)(0x08, 0x08, 0x00, 0x01)

# set DIO channel 0, operating as SPI CS to low (0)
dwf.FDwfDigitalSpiSelect(hdwf, c_int(0), c_int(0))

# perform a SPI write operation over DQ line 0, with 8 bits per word, sending the above buffer (b)
dwf.FDwfDigitalSpiWrite(hdwf, c_int(0), c_int(8), b, c_int(len(b)))

# set DIO channel 0, operating as SPI CS to high (1)
dwf.FDwfDigitalSpiSelect(hdwf, c_int(0), c_int(1))
time.sleep(.05)

try:
    while True:
        # begin analog in acquisition
        dwf.FDwfAnalogInConfigure(hdwf, c_int(1), c_int(1))

        b = (c_ubyte*2)(0x40,0x00)
        r = (c_ubyte*2)()

        # set DIO channel 0, operating as SPI CS to low (0)
        dwf.FDwfDigitalSpiSelect(hdwf, c_int(0), c_int(0))
        
        # perform a SPI write/read operation of the status register
        dwf.FDwfDigitalSpiWriteRead(hdwf, c_int(1), c_int(8), b, c_int(len(b)), r, c_int(len(r)))

        # set DIO channel 0, operating as SPI CS to high (1)
        dwf.FDwfDigitalSpiSelect(hdwf, c_int(0), c_int(1))

        # check the ~RDY bit and if set then continue waiting if it isn't
        if ((r[1] >> 7 & 1) == 1):
            print("Conversion not ready")
            continue

        # check the ERR and halt if encountered
        if ((r[1] >> 6 & 1) == 1):
            print("An error in conversion was encountered")
            break

        # issue a read and calculate the temperature from the data
        b = (c_ubyte*4)(0x58,0x00,0x00,0x00)
        r = (c_ubyte*4)()

        # set DIO channel 0, operating as SPI CS to low (0)
        dwf.FDwfDigitalSpiSelect(hdwf, c_int(0), c_int(0))
        
        # perform a SPI write/read operation using MOSI/MISO with 8 bit words, using b to send and r to receive
        dwf.FDwfDigitalSpiWriteRead(hdwf, c_int(1), c_int(8), b, c_int(len(b)), r, c_int(len(b)))

        # set DIO channel 0, operating as SPI CS to high (1)
        dwf.FDwfDigitalSpiSelect(hdwf, c_int(0), c_int(1))
        
        # convert byte array into numeric value
        ch0Data = 0
        for i in range(1, 4):
            ch0Data = ch0Data << 8 | r[i]

        # complete the acquisition & average the samples
        status = c_byte()
        while True:
            # cgeck the acquisiton status, and break the loop if done, otherwise pause and try again
            dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(status))
            if status.value == DwfStateDone.value:
                break
            time.sleep(0.1)

        print("Acquisiton Done")
        samples = (c_double*4000)()
        # read the acquired samples into samples
        dwf.FDwfAnalogInStatusData(hdwf, 0, samples, 4000)

        mVref = 2.5
        PGAGain = 1

        # convert the data into a numeric voltage value
        thermoVoltage = (((ch0Data / 8388608) - 1.0) * (mVref / PGAGain))

        # find the average of the samples
        avg = sum(samples)/len(samples)
        
        print("Measured %f mV at the thermocouple junction, and got %f mV from the ADC" % (avg * 1000, thermoVoltage * 1000))

        time.sleep(5)

except KeyboardInterrupt:
    pass

print("\ncleaning up...")
dwf.FDwfDeviceCloseAll()