from time import sleep
from WF_SDK import device, scope, tools
from WF_SDK.protocol import uart
import numpy as np

def sendStringUART(dev, section):
    i = 0
    while i < 8:
        if i < len(section):
            uart.write(dev, section[i])
        else:
            uart.write(dev, "\0")
        i += 1
    # uart.write(dev, section[i])

def FFT(buffer, freq_sweep=[0, 100e3]):
    """
    Compute single-sided magnitude spectrum and frequency vector (in MHz)
    buffer : iterable of voltage samples (float)
    freq_sweep : [start_freq, stop_freq] in Hz (only for frequency cropping)
    Returns: (spectrum_magnitude, frequency_mhz_array)
    """
    # convert to numpy array
    x = np.asarray(buffer, dtype=float)
    N = x.size
    if N == 0:
        return np.array([]), np.array([])

    # Sampling frequency is in scope.data.sampling_frequency (Hz)
    fs = scope.data.sampling_frequency

    # compute FFT
    X = np.fft.rfft(x * np.hanning(N))   # window to reduce leakage (Hann)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)  # Hz

    # magnitude (abs) and optionally normalize (divide by N)
    mag = np.abs(X) / N

    # Crop to requested freq_sweep range
    start_freq = float(freq_sweep[0])
    stop_freq = float(freq_sweep[1])
    mask = (freqs >= start_freq) & (freqs <= stop_freq)
    freqs_crop = freqs[mask] / 1e6   # convert to MHz for your existing code
    mag_crop = mag[mask]

    return mag_crop, freqs_crop

# ------------------- USER SETTINGS -------------------
PIN_TX = 0           # DIO pin used for UART TX (ADP3450 DIO0)
PIN_RX = 1           # optional RX pin if you want to read back
BAUDRATE = 115200
FREQ = [100, 200, 300, 400, 500]
CMD = ""
# ------------------------------------------------------

print("Opening ADP3450 device...")
dev = device.open()      # open first connected device
max_buf = dev.analog.input.max_buffer_size

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
print("Max buffer size: ", max_buf)

# Main send loop
try:
    while True:
        CMD = input("\nEnter desired frequency: ")
        msg = CMD
        sendStringUART(dev, msg)
        sleep(1)
        while True:
            mainloop = False
            RES = bytes(uart.read(dev))
            if RES.decode("utf-8") == "Received":
                mainloop = True
                time = []
                print(f"\nMeasuring EIS at {CMD.strip()} Hz...")
                # initialize the scope with default settings
                # choose sensible values
                samp_freq = int(100*float(CMD))       # 1 MHz sampling
                buf_size = 300
                scope.open(dev, sampling_frequency=samp_freq, buffer_size=buf_size, offset=0, amplitude_range=5)
                sleep(1)

                current = scope.record(dev, channel=1)
                volt_1 = scope.record(dev, channel=2)

                I_FFT_abs, I_FFT_freq = FFT(current, freq_sweep = [0, 100e3])
                V1_FFT_abs, V1_FFT_freq = FFT(volt_1, freq_sweep = [0, 100e3])

                # generate buffer for time moments
                # for index in range(len(current)):
                #     time.append(index * 1e03 / scope.data.sampling_frequency)
                print("Impedance magnitude in ohms: ", V1_FFT_abs[V1_FFT_freq == int(CMD)*1e6]/I_FFT_abs[I_FFT_freq == int(CMD)*1e6])
                sleep(.5)
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