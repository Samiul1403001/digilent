from MyDigilent import MyDigilent, clean_buffer, freq_selection_signal
from time import sleep
import numpy as np

FREQ = [1, 5, 10, 20]

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
                print(f"\nMeasuring EIS at {CMD.strip()} Hz...")

                # initialize the scope with default settings
                # choose sensible values
                buffer_size = f * 100
                sample_rate = int(100*float(CMD))

                data_sets = Digi_1.scope_record(sample_rate, buffer_size)

            if RES.decode("utf-8") == "DoneRecv":
                I_freq = freq_selection_signal(100*(data_sets[0]-np.mean(data_sets[0])), freq_sweep=[f*0.8, f*1.2], sample_rate=sample_rate)
                V_freq = freq_selection_signal(data_sets[1]-np.mean(data_sets[1]), freq_sweep=[f*0.8, f*1.2], sample_rate=sample_rate)

                sfreq = V_freq
                print(sfreq)
                if sfreq == None:
                    sfreq = f
                
                I_clean, Iparams = clean_buffer(100*(data_sets[0]-np.mean(data_sets[0])), signal_freq=sfreq, sample_rate=sample_rate)
                V_clean, Vparams = clean_buffer(data_sets[1]-np.mean(data_sets[1]), signal_freq=sfreq, sample_rate=sample_rate)

                print(f"Recovered Amplitudes: V: {Vparams[0]:.3f}, I: {Iparams[0]:.3f}")
                print(f"Recovered Phases: V: {Vparams[1]*180/np.pi:.3f}, I: {Iparams[1]*180/np.pi:.3f}")

                I_FFT_real = Iparams[0] * np.cos(Iparams[1])
                I_FFT_imag = Iparams[0] * np.sin(Iparams[1])
                V1_FFT_real = Vparams[0] * np.cos(Vparams[1])
                V1_FFT_imag = Vparams[0] * np.sin(Vparams[1])

                V_comp = V1_FFT_real + 1j * V1_FFT_imag
                I_comp = I_FFT_real + 1j * I_FFT_imag

                Z = (V_comp / I_comp)
                print("Impedance in ohms: " + str(Z.real) + "+(" + str(Z.imag) + "j)")
                sample[i, 0] = sfreq
                sample[i, 1] = Z.real
                sample[i, 2] = -Z.imag
                i+=1
                mainloop = False
                print(f"\nMeasuring EIS at {CMD.strip()} Hz is done.")
    rows_to_keep = ~ (sample == 0).all(axis=1)
    sample = sample[rows_to_keep]

    Digi_1.close()
    print("Device closed.")

except KeyboardInterrupt:
    Digi_1.close()
    print("Device closed.")