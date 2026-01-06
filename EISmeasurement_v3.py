from MyDigilent import MyDigilent, FFT, clean_buffer, freq_selection_signal
from time import sleep
import numpy as np

seed_freq = [1,0.8,0.65,0.5,0.4,0.3,0.25,0.2,0.15,0.125]
FREQ = []

for i in range(1, -2, -1):
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
                elif (f >= 1) and (f <= 5):
                    buffer_size = 500
                else:
                    buffer_size = 1000
                
                sample_rate = int(100*float(CMD))

                data_sets = Digi_1.scope_record(sample_rate, buffer_size)

            if RES.decode("utf-8") == "DoneRecv":
                I_freq = freq_selection_signal(100*(data_sets[0]-np.mean(data_sets[0])), freq_sweep=[f*0.8, f*1.2], sample_rate=sample_rate)
                V_freq = freq_selection_signal(data_sets[1]-np.mean(data_sets[1]), freq_sweep=[f*0.8, f*1.2], sample_rate=sample_rate)

                sfreq = I_freq
                print(sfreq)
                if sfreq == None:
                    sfreq = f
                
                I_clean, Iparams = clean_buffer(100*(data_sets[0]-np.mean(data_sets[0])), signal_freq=sfreq, sample_rate=sample_rate)
                V1_clean, V1params = clean_buffer(data_sets[1]-np.mean(data_sets[1]), signal_freq=sfreq, sample_rate=sample_rate)

                I_FFT_freqs, I_FFT_abs, I_FFT_real, I_FFT_imag = FFT(-I_clean, freq_sweep=[sfreq*0.2, sfreq*2], sample_rate=sample_rate)
                V1_FFT_freqs, V1_FFT_abs, V1_FFT_real, V1_FFT_imag = FFT(V1_clean, freq_sweep=[sfreq*0.2, sfreq*2], sample_rate=sample_rate)

                Ifreq_mask = (I_FFT_freqs >= sfreq - sfreq*0.5) & (I_FFT_freqs <= sfreq + sfreq*0.5)
                V1freq_mask = (V1_FFT_freqs >= sfreq - sfreq*0.5) & (V1_FFT_freqs <= sfreq + sfreq*0.5)
                Iidx = np.argmax(I_FFT_abs[Ifreq_mask])
                V1idx = np.argmax(V1_FFT_abs[V1freq_mask])

                print(f"Recovered Amplitudes: V: {V1_FFT_abs[V1idx]:.3E}, I: {I_FFT_abs[Iidx]:.3E}")
                print(f"Recovered frequency: F: {I_FFT_freqs[Iidx]:.6f}")
                V_comp = V1_FFT_real[V1idx] + 1j * V1_FFT_imag[V1idx]
                I_comp = I_FFT_real[Iidx] + 1j * I_FFT_imag[Iidx]
                Z = V_comp / I_comp
                print("Impedance in ohms: " + str(Z.real) + "+(" + str(Z.imag) + "j)")

                sample[i, 0] = np.round(I_FFT_freqs[Iidx], decimals=6)
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