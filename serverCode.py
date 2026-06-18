from MyDigilent import MyDigilent, freq_selection_signal, dual_phase_demod, FFT, fir_bandpass, correct
from time import sleep
import numpy as np, socket, struct, mlrepo as ml

# SoH estimator
SoH_est = ml.NumpySimpleSoHLSTM('simple_soh_weights.npz')

# --- TCP Configuration ---
# 1. SERVER CONFIG: Listen on ALL interfaces
TCP_IP = '0.0.0.0'
TCP_PORT = 5005            

# --- Hardware Initialization ---
PIN_TX = 0            
PIN_RX = 1            
BAUDRATE = 115200

# Initialize Hardware ONCE (before entering the network loop)
print("Initializing Digilent-ADP3450...")
Digi_1 = MyDigilent(tx=PIN_TX, rx=PIN_RX, baud_rate=BAUDRATE, parity="none", data_bits=8, stop_bits=1)
Digi_1.scope_setup(channels=[1, 2, 3, 4])
sleep(1)

max_buf = Digi_1.dev.analog.input.max_buffer_size
fsample_max = 1e6
print(f"Max buffer size per channel: {max_buf}, Max sampling rate: {fsample_max}")

# --- Frequency Setup ---
f_freq = [10e3, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2]
finit_idx = 3
fperdecade = 10
FREQ_TEMPLATE = [] 
FREQ_TEMPLATE.append(f_freq[finit_idx])
for i in range(finit_idx, len(f_freq)-1):
    for k in range(1, fperdecade+1):
        FREQ_TEMPLATE.append(10**(np.log10(f_freq[i]).item()-k/fperdecade))

# --- MAIN SERVER LOOP ---
server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

try:
    server_sock.bind((TCP_IP, TCP_PORT))
    server_sock.listen(1)
    print(f"Server listening on {TCP_IP}:{TCP_PORT}")
    print("Digilent is ready. Waiting for Client connection...")

    # 2. OUTER LOOP: Keeps the server alive forever
    while True:
        try:
            # Block here until a Client connects
            conn, addr = server_sock.accept()
            print(f"\nClient connected from: {addr}")
            
            client_connected = True

            # 3. CLIENT SESSION LOOP
            while client_connected:
                print("--- STANDBY: Waiting for 'START' command ---")
                
                # Ensure blocking mode while waiting for command
                conn.setblocking(True) 
                
                try:
                    command_raw = conn.recv(1024)
                    if not command_raw:
                        print("Client disconnected.")
                        client_connected = False
                        break 
                    
                    command = command_raw.decode('utf-8').strip()
                except ConnectionResetError:
                    print("Connection reset by peer.")
                    client_connected = False
                    break
                except Exception as e:
                    print(f"Receive Error: {e}")
                    client_connected = False
                    break

                if command == "START":
                    print("START received. Beginning Measurement Sequence...")
                    
                    # Initialize run variables
                    sample_c1 = np.zeros([31, 6])
                    sample_c2 = np.zeros([31, 6])
                    sample_c3 = np.zeros([31, 6])
                    i_idx = 0
                    stop_requested = False

                    # Loop through frequencies
                    for f in FREQ_TEMPLATE:
                        
                        # --- CHECK FOR STOP COMMAND (Non-Blocking) ---
                        try:
                            conn.setblocking(False) # Peek using the CLIENT connection
                            cmd_check = conn.recv(1024) 
                            if cmd_check and "STOP" in cmd_check.decode('utf-8'):
                                print("\n!!! STOP command received. Halting measurement !!!")
                                stop_requested = True
                                conn.setblocking(True)
                                break 
                        except BlockingIOError:
                            pass 
                        except Exception as e:
                            print(f"Socket error during check (Client likely disconnected): {e}")
                            stop_requested = True
                            client_connected = False
                            break
                        
                        conn.setblocking(True) # Restore blocking
                        # ---------------------------------------------

                        # Hardware Logic (Same as before)
                        CMD = str(f)
                        Digi_1.sendStringUART(CMD)
                        sleep(1) 
                        
                        mainloop = True
                        while mainloop:
                            RES = bytes(Digi_1.uart_read())
                            res_str = RES.decode("utf-8")

                            if res_str == "Received":
                                print(f"Measuring EIS at {CMD.strip()} Hz...")
                                buffer_size = int(max_buf)
                                sample_rate = int(fsample_max)
                                ncycle = int(buffer_size/(sample_rate/f))
                                if f < 0.1:
                                    ncycle = 2
                                    sample_rate = int(buffer_size / (ncycle / f))
                                    
                                elif f <= 10 and f >= 0.1:
                                    ncycle = int(7.5*np.log10(f)+12.5)
                                    sample_rate = int(buffer_size / (ncycle / f))

                                else:
                                    est_ncycle = int(0.6228 * np.exp(2.2101*np.log10(f)))
                                    while (ncycle < est_ncycle):
                                        sample_rate = int(sample_rate * 0.9)
                                        ncycle = int(buffer_size/(sample_rate/f))
                                    if ncycle < 2:
                                        ncycle = 2
                                        sample_rate = int(f*buffer_size/ncycle)
                                data_sets = Digi_1.scope_record(sample_rate, buffer_size)
                                print(f"buffer size: {buffer_size}, Perturbation freq: {f}, Sampling frequency: {sample_rate}, Number of cycles: {ncycle}")

                            elif res_str == "DoneRecv":
                                # Calculation Logic
                                Imeas = (data_sets[0]-np.mean(data_sets[0]))/0.033
                                V1meas = data_sets[1]-np.mean(data_sets[1])
                                V2meas = data_sets[2]-np.mean(data_sets[2])
                                V3meas = data_sets[3]-np.mean(data_sets[3])

                                Imeas_filtered = fir_bandpass(Imeas, sample_rate, f*0.8, f*1.2)
                                V1meas_filtered = fir_bandpass(V1meas, sample_rate, f*0.8, f*1.2)
                                V2meas_filtered = fir_bandpass(V2meas, sample_rate, f*0.8, f*1.2)
                                V3meas_filtered = fir_bandpass(V3meas, sample_rate, f*0.8, f*1.2)

                                rng_int = 1 / 10 ** int(-np.log10(f) + 3)

                                if rng_int < 0.001:
                                    I_freq = freq_selection_signal(Imeas_filtered, freq_sweep=[f*0.998, f*1.002], sample_rate=sample_rate)
                                else:
                                    _, _, _, _, I_freq = FFT(Imeas_filtered, freq_sweep=[f*(1-rng_int), f*(1+rng_int)], sample_rate=sample_rate)

                                sfreq = I_freq if I_freq is not None else f
                                
                                Iamp, Iphase = dual_phase_demod(Imeas_filtered, sfreq, sample_rate)
                                V1amp, V1phase = dual_phase_demod(V1meas_filtered, sfreq, sample_rate)
                                V2amp, V2phase = dual_phase_demod(V2meas_filtered, sfreq, sample_rate)
                                V3amp, V3phase = dual_phase_demod(V3meas_filtered, sfreq, sample_rate)

                                print(f"Freq: {sfreq:.5f} Hz | V_amp: {V2amp:.2E} | I_amp: {Iamp:.2E}")

                                I_real = Iamp * np.cos(Iphase+np.pi)
                                I_imag = Iamp * np.sin(Iphase+np.pi)
                                V1_real = V1amp * np.cos(V1phase)
                                V1_imag = V1amp * np.sin(V1phase)
                                V2_real = V2amp * np.cos(V2phase)
                                V2_imag = V2amp * np.sin(V2phase)
                                V3_real = V3amp * np.cos(V3phase)
                                V3_imag = V3amp * np.sin(V3phase)

                                V1_comp = V1_real + 1j * V1_imag
                                V2_comp = V2_real + 1j * V2_imag
                                V3_comp = V3_real + 1j * V3_imag
                                I_comp = I_real + 1j * I_imag
                                Z1 = (V1_comp / I_comp)
                                Z2 = (V2_comp / I_comp)
                                Z3 = (V3_comp / I_comp)

                                # Data Quality Check
                                if i_idx > 0 and ((Z1.real < 0.98*sample_c1[i_idx-1, 1] and Z1.real < 0) or (Z2.real < 0.98*sample_c2[i_idx-1, 1] and Z2.real < 0) or (Z3.real < 0.98*sample_c3[i_idx-1, 1] and Z3.real < 0)):
                                    print("\nFrequency skipped (Impedance Drop)...\n")
                                    break

                                Z1real, Z1imag = correct(sfreq, Z1.real, -Z1.imag)
                                print("Cell-1 Impedance: " + str(Z1real) + "+(" + str(-Z1imag) + "j)")
                                Z2real, Z2imag = correct(sfreq, Z2.real, -Z2.imag)
                                print("Cell-2 Impedance: " + str(Z2real) + "+(" + str(-Z2imag) + "j)")
                                Z3real, Z3imag = correct(sfreq, Z3.real, -Z3.imag)
                                print("Cell-3 Impedance: " + str(Z3real) + "+(" + str(-Z3imag) + "j)")
                                
                                sample_c1[i_idx, 0] = np.mean(data_sets[1])
                                sample_c1[i_idx, 1] = np.log10(sfreq)
                                sample_c1[i_idx, 2] = Z1real
                                sample_c1[i_idx, 3] = Z1imag
                                sample_c1[i_idx, 4] = np.abs(Z1real - 1j * Z1imag)
                                sample_c1[i_idx, 5] = np.angle(Z1real - 1j * Z1imag, deg=True)

                                sample_c2[i_idx, 0] = np.mean(data_sets[2])
                                sample_c2[i_idx, 1] = np.log10(sfreq)
                                sample_c2[i_idx, 2] = Z2real
                                sample_c2[i_idx, 3] = Z2imag
                                sample_c2[i_idx, 4] = np.abs(Z2real - 1j * Z2imag)
                                sample_c2[i_idx, 5] = np.angle(Z2real - 1j * Z2imag, deg=True)

                                sample_c3[i_idx, 0] = np.mean(data_sets[3])
                                sample_c3[i_idx, 1] = np.log10(sfreq)
                                sample_c3[i_idx, 2] = Z3real
                                sample_c3[i_idx, 3] = Z3imag
                                sample_c3[i_idx, 4] = np.abs(Z3real - 1j * Z3imag)
                                sample_c3[i_idx, 5] = np.angle(Z3real - 1j * Z3imag, deg=True)

                                # --- ML based SoH estimation ---
                                output_c1 = SoH_est.predict(sample_c1.reshape(1, 6, 31).astype(np.float32))
                                print(f"\n\nThe estimated SoH of cell-1 is: {str(np.round(output_c1*100, decimals=2))}%\n")
                                output_c2 = SoH_est.predict(sample_c2.reshape(1, 6, 31).astype(np.float32))
                                print(f"\n\nThe estimated SoH of cell-2 is: {str(np.round(output_c2*100, decimals=2))}%\n")
                                output_c3 = SoH_est.predict(sample_c3.reshape(1, 6, 31).astype(np.float32))
                                print(f"\n\nThe estimated SoH of cell-3 is: {str(np.round(output_c3*100, decimals=2))}%\n")
                                
                                # --- Send Data to Host ---
                                try:
                                    # 1. Extract the current row for each cell (6 values each)
                                    row_c1 = sample_c1[i_idx, :].flatten()
                                    row_c2 = sample_c2[i_idx, :].flatten()
                                    row_c3 = sample_c3[i_idx, :].flatten()
                                    
                                    # 2. Format the SoH estimates (clip, scale to %, round, and flatten)
                                    soh_1 = np.round(np.clip(output_c1, 0, 1) * 100, decimals=2).flatten()
                                    soh_2 = np.round(np.clip(output_c2, 0, 1) * 100, decimals=2).flatten()
                                    soh_3 = np.round(np.clip(output_c3, 0, 1) * 100, decimals=2).flatten()
                                    
                                    # 3. Concatenate everything into one flat array
                                    # Total length will be (6 + 1) * 3 = 21 elements
                                    combined_data = np.concatenate([
                                        row_c1, soh_1, 
                                        row_c2, soh_2, 
                                        row_c3, soh_3
                                    ])
                                    
                                    # 4. Convert to bytes (ensure consistent float64 type for struct unpacking)
                                    data_bytes = combined_data.astype(np.float64).tobytes()
                                    header = struct.pack('>I', len(data_bytes))
                                    
                                    # 5. Send over TCP
                                    conn.sendall(header + data_bytes)
                                    print(f"Sent combined measurements (21 values) to Client.")
                                    
                                except Exception as e:
                                    print(f"Send failed (Client disconnected?): {e}")
                                    client_connected = False
                                    stop_requested = True # Force loop exit

                                i_idx += 1
                                mainloop = False 
                                sleep(int(3*(3-np.log10(sfreq))))

                        if not client_connected: break

                    print("Sequence finished or stopped.")
                    # Data saving logic (optional)...

            # End of Client Session
            if conn: conn.close()
            print("Session ended. Returning to wait state...\n")

        except Exception as e:
            print(f"Server Loop Error: {e}")
            # Ensure we don't crash the server script
            try: conn.close()
            except: pass

except KeyboardInterrupt:
    print("\nServer shutting down manually.")
finally:
    Digi_1.close()
    server_sock.close()
    print("Device and Socket closed.")