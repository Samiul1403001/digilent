from MyDigilent import MyDigilent, FFT
from time import sleep
import numpy as np

seed_freq = [1,0.8,0.65,0.5,0.4,0.3,0.25,0.2,0.15,0.125]
FREQ = []

for i in range(0, -1, -1):
    freq = seed_freq*10
    FREQ.extend([item * 10**i for item in seed_freq])

FREQ = np.round(FREQ, decimals=4)

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
                print(f"\nMeasuring EIS at {CMD.strip()} Hz...")

                # initialize the scope with default settings
                # choose sensible values
                if f < 1:
                    buffer_size = 200
                elif (f >= 1) and (f <=10):
                    buffer_size = 500
                else:
                    buffer_size = 1000
                
                sample_rate = int(100*float(CMD))

                data_sets = Digi_1.scope_record(sample_rate, buffer_size)

            if RES.decode("utf-8") == "DoneRecv":
                I_FFT_freqs, I_FFT_abs, I_FFT_real, I_FFT_imag = FFT(-100*(data_sets[0]-np.mean(data_sets[0])), freq_sweep=[f*0.8, f*1.2], sample_rate=sample_rate)
                V1_FFT_freqs, V1_FFT_abs, V1_FFT_real, V1_FFT_imag = FFT(data_sets[1]-np.mean(data_sets[1]), freq_sweep=[f*0.8, f*1.2], sample_rate=sample_rate)

                Ifreq_mask = (I_FFT_freqs >= f - f*0.05) & (I_FFT_freqs <= f + f*0.05)
                V1freq_mask = (V1_FFT_freqs >= f - f*0.05) & (V1_FFT_freqs <= f + f*0.05)
                Iidx = np.argmax(I_FFT_abs[Ifreq_mask])
                V1idx = np.argmax(V1_FFT_abs[V1freq_mask])

                print(f"Recovered Amplitudes: V: {V1_FFT_abs[V1idx]:.3E}, I: {I_FFT_abs[Iidx]:.3E}")
                V_comp = V1_FFT_real[V1idx] + 1j * V1_FFT_imag[V1idx]
                I_comp = I_FFT_real[Iidx] + 1j * I_FFT_imag[Iidx]
                Z = V_comp / I_comp
                print("Impedance in ohms: " + str(Z.real) + "+(" + str(Z.imag) + "j)")

                sample[i, 0] = np.round(I_FFT_freqs[Iidx], decimals=4)
                sample[i, 1] = Z.real
                sample[i, 2] = -Z.imag
                i+=1
                mainloop = False
                print(f"\nMeasuring EIS at {CMD.strip()} Hz is done.")
        sleep(3)
    rows_to_keep = ~ (sample == 0).all(axis=1)
    sample = sample[rows_to_keep]

    np.savetxt("EIS_Data/Test-1.csv", sample, delimiter=',')
    print("Data has been saved...")

    Digi_1.close()
    print("Device closed.")

except KeyboardInterrupt:
    Digi_1.close()
    print("Device closed.")