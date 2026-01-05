from MyDigilent import MyDigilent, clean_buffer, freq_selection_signal
from time import sleep
import numpy as np

FREQ = [#10000,7943.282227,6309.573242,5011.87207,3981.071289,3162.277344,2511.88623,1995.262085,1584.892944,1258.925171,
        #999.9997559,794.3280029,630.9571533,501.1870728,398.1070251,316.2276306,251.1885223,199.526123,158.4892273,125.8924637,
        99.99993896,79.43276978,63.09569168,50.11868668,39.81068802,31.62275314,25.11884499,19.95260811,15.84891987,12.58924389,
        9.999991417,7.943275452,6.309567928,5.011868,3.981068134,3.162274837,2.511884212,1.995260477,1.584891677,1.258924127,
        0.999998927,0.794327378,0.63095665,0.501186669,0.398106724,0.316227406,0.251188338,0.199525982,0.158489123,0.125892386#,
        #0.099999875,0.079432718,0.063095652,0.050118655,0.039810661,0.03162273,0.025118826,0.019952592,0.015848907,0.012589234,0.01
        ] 

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

                I_FFT_real = Iparams[0] * np.cos(Iparams[1]+np.pi)
                I_FFT_imag = Iparams[0] * np.sin(Iparams[1]+np.pi)
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

    np.savetxt("EIS_Data/Test-1.csv", sample, delimiter=',')
    print("Data has been saved...")

    Digi_1.close()
    print("Device closed.")

except KeyboardInterrupt:
    Digi_1.close()
    print("Device closed.")