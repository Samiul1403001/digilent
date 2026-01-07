from MyDigilent import MyDigilent, freq_selection_signal, dual_phase_demod
from time import sleep
import numpy as np, socket, struct

# TCP configuration
TCP_IP = '10.115.78.142'  # Localhost (use Host PC IP if running in ADP3450 Linux Mode)
TCP_PORT = 5005           # Arbitrary port (must match receiver)

print(f"Connecting to server at {TCP_IP}:{TCP_PORT}...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    sock.connect((TCP_IP, TCP_PORT))
    print("Connected!")
except ConnectionRefusedError:
    print("Failed to connect. Is the Receiver script running?")
    quit()

# Initialization
f_freq = [1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2]
finit_idx = 1
fperdecade = 10

FREQ = []
FREQ.append(f_freq[finit_idx])
for i in range(finit_idx, finit_idx+3):
    for k in range(1, fperdecade+1):
        FREQ.append(10**(np.log10(f_freq[i]).item()-k/fperdecade))

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
        sleep(1)
        mainloop = True
        while mainloop == True:
            RES = bytes(Digi_1.uart_read())
            if RES.decode("utf-8") == "Received":
                print(f"\nMeasuring EIS at {CMD.strip()} Hz...")

                # initialize the scope with default settings
                # choose sensible values
                if f < 1:
                    buffer_size = 180
                elif (f >= 1) and (f <= 5):
                    buffer_size = 450
                else:
                    buffer_size = 900
                
                sample_rate = int(100*float(CMD))

                data_sets = Digi_1.scope_record(sample_rate, buffer_size)

            if RES.decode("utf-8") == "DoneRecv":
                Imeas = 100*(data_sets[0]-np.mean(data_sets[0]))
                V1meas = data_sets[1]-np.mean(data_sets[1])

                I_freq = freq_selection_signal(Imeas, freq_sweep=[f*0.9, f*1.1], sample_rate=sample_rate)
                V1_freq = freq_selection_signal(V1meas, freq_sweep=[f*0.9, f*1.1], sample_rate=sample_rate)

                sfreq = I_freq
                print(sfreq)
                if sfreq == None:
                    sfreq = f
                
                Iamp, Iphase = dual_phase_demod(Imeas, sfreq, sample_rate)
                V1amp, V1phase = dual_phase_demod(V1meas, sfreq, sample_rate)

                
                print(f"Recovered Amplitudes: V: {V1amp:.3E}, I: {Iamp:.3E}")
                print(f"Recovered Phases: V: {V1phase*180/np.pi:.3f}, I: {(Iphase+np.pi)*180/np.pi:.3f}")

                I_real = Iamp * np.cos(Iphase+np.pi)
                I_imag = Iamp * np.sin(Iphase+np.pi)
                V1_real = V1amp * np.cos(V1phase)
                V1_imag = V1amp * np.sin(V1phase)

                V_comp = V1_real + 1j * V1_imag
                I_comp = I_real + 1j * I_imag

                Z = (V_comp / I_comp)
                print("Impedance in ohms: " + str(Z.real) + "+(" + str(Z.imag) + "j)")
                if i > 0 and Z.real < 0.9*sample[i-1, 1]:
                    print("\nFrequency skipped...\n")
                    break

                sample[i, 0] = sfreq
                sample[i, 1] = Z.real
                sample[i, 2] = -Z.imag
                i+=1
                mainloop = False
                print(f"\nMeasuring EIS at {CMD.strip()} Hz is done.")
                
                # A. Serialize data: Convert float array to bytes
                data_bytes = sample.flatten().tobytes()

                # B. Pack the size (Integer, 4 bytes, Big Endian)
                header = struct.pack('>I', len(data_bytes))

                # C. Send Header + Data
                sock.sendall(header + data_bytes)
                print(f"Sent {sample.shape[0]} samples to the host.")
        
        sleep(5)
    rows_to_keep = ~ (sample == 0).all(axis=1)
    sample = sample[rows_to_keep]

    np.savetxt("EIS_Data/Test-1.csv", sample, delimiter=',')
    print("Data has been saved...")

    Digi_1.close()
    sock.close()
    print("Device closed.")

except KeyboardInterrupt:
    Digi_1.close()
    sock.close()
    print("Device closed.")