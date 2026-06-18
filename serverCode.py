from MyDigilent import MyDigilent, freq_selection_signal, dual_phase_demod, FFT, fir_bandpass, HolderCalibrator
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

# Calibrator
_ref = np.array([
    [10000,       0.033761269, -0.042496864],
    [7943.282227, 0.03220302,  -0.034446557],
    [6309.573242, 0.030926482, -0.027641812],
    [5011.87207,  0.030036374, -0.021977292],
    [3981.071289, 0.029535777, -0.017348838],
    [3162.277344, 0.029305876, -0.013586192],
    [2511.88623,  0.029191959, -0.01051091 ],
    [1995.262085, 0.029165459, -0.008006643],
    [1584.892944, 0.029248144, -0.005966271],
    [1258.925171, 0.02939431,  -0.004356155],
    [999.9997559, 0.029571059, -0.003080103],
    [794.3280029, 0.02978809,  -0.002058299],
    [630.9571533, 0.030044151, -0.001215238],
    [501.1870728, 0.030289148, -0.000552918],
    [398.1070251, 0.03056527,  -2.69e-05   ],
    [316.2276306, 0.03086086,   0.00039632 ],
    [251.1885223, 0.031104502,  0.000743486],
    [199.526123,  0.031407376,  0.001034341],
    [158.4892273, 0.031683898,  0.001271967],
    [125.8924637, 0.031965693,  0.001472903],
    [99.99993896, 0.032257054,  0.001654974],
    [79.43276978, 0.032544226,  0.001827445],
    [63.09569168, 0.032825113,  0.001982701],
    [50.11868668, 0.033102187,  0.002147839],
    [39.81068802, 0.033398963,  0.002286798],
    [31.62275314, 0.033682589,  0.002483291],
    [25.11884499, 0.033936066,  0.00269303 ],
    [19.95260811, 0.034209742,  0.002950715],
    [15.84891987, 0.034452543,  0.003288451],
    [12.58924389, 0.034744094,  0.003710181],
    [9.999991417, 0.034935293,  0.004419803],
    [7.943275452, 0.035249442,  0.004930139],
    [6.309567928, 0.035511191,  0.00590276 ],
    [5.011868,    0.035874002,  0.006914585],
    [3.981068134, 0.036540877,  0.008372373],
    [3.162274837, 0.037235424,  0.010099115],
    [2.511884212, 0.038103903,  0.012052846],
    [1.995260477, 0.039012379,  0.014414539],
    [1.584891677, 0.040390322,  0.017595442],
    [1.258924127, 0.042754322,  0.020923984],
    [0.999998927, 0.045052756,  0.024751675],
    [0.794327378, 0.049111293,  0.029261379],
    [0.63095665,  0.053711905,  0.033747515],
    [0.501186669, 0.060209005,  0.038281623],
    [0.398106724, 0.067694566,  0.042015018],
    [0.316227406, 0.076640895,  0.044529143],
    [0.251188338, 0.086032085,  0.045692313],
    [0.199525982, 0.095435843,  0.045059896],
    [0.158489123, 0.104268019,  0.043054309],
    [0.125892386, 0.111988447,  0.040141383],
    [0.099999875, 0.118488608,  0.036748339],
    [0.079432718, 0.12393687,   0.033327328],
    [0.063095652, 0.128621609,  0.030293856],
    [0.050118655, 0.132608593,  0.027946293],
    [0.039810661, 0.136193226,  0.026147991],
    [0.03162273,  0.139570133,  0.024914932],
    [0.025118826, 0.14299696,   0.024290465],
    [0.019952592, 0.146547858,  0.02446115 ],
    [0.015848907, 0.150169768,  0.025386908],
    [0.012589234, 0.153866438,  0.026937101],
    [0.01,        0.157800657,  0.02894049 ]
])
_meas_holder1 = np.array([
    [10.0,         0.064310753, 0.003780446],
    [7.943145752,  0.064542767, 0.004498564],
    [6.309448242,  0.065131378, 0.005358283],
    [5.01159668,   0.065125686, 0.006124746],
    [3.980743408,  0.065565601, 0.007402311],
    [3.162109375,  0.066377838, 0.008774717],
    [2.511749268,  0.067277033, 0.011030777],
    [1.995239258,  0.068308829, 0.013284014],
    [1.584655762,  0.069334461, 0.015910647],
    [1.25881958,   0.07112493,  0.018925738],
    [0.999755859,  0.073844257, 0.022281353],
    [0.794250488,  0.077403913, 0.026088321],
    [0.630767822,  0.081645948, 0.030617366],
    [0.501098633,  0.087681849, 0.032949235],
    [0.397979736,  0.094769932, 0.036607205],
    [0.316162109,  0.102815928, 0.038563228],
    [0.250976563,  0.109689527, 0.039356409],
    [0.199523926,  0.118074906, 0.037907794],
    [0.15838623,   0.125436013, 0.035678454],
    [0.12588501,   0.131372822, 0.033477382],
    [0.09999,      0.137873613, 0.030135885],
    [0.079439972,  0.141376224, 0.027461742],
    [0.063101413,  0.144377342, 0.024434176],
    [0.050123234,  0.147335978, 0.022420025],
    [0.039814698,  0.148012276, 0.020404185],
    [0.031625623,  0.150982504, 0.020178247],
    [0.025121125,  0.150792897, 0.019329331],
    [0.019954618,  0.153867789, 0.020487372],
    [0.015850358,  0.15635527,  0.01896543],
    [0.012590513,  0.158453721, 0.022077421],
    [0.01,         0.159896748, 0.024543966]
])
_meas_holder2 = np.array([
    [10.0,         0.064310753, 0.003780446],
    [7.943145752,  0.064542767, 0.004498564],
    [6.309448242,  0.065131378, 0.005358283],
    [5.01159668,   0.065125686, 0.006124746],
    [3.980743408,  0.065565601, 0.007402311],
    [3.162109375,  0.066377838, 0.008774717],
    [2.511749268,  0.067277033, 0.011030777],
    [1.995239258,  0.068308829, 0.013284014],
    [1.584655762,  0.069334461, 0.015910647],
    [1.25881958,   0.07112493,  0.018925738],
    [0.999755859,  0.073844257, 0.022281353],
    [0.794250488,  0.077403913, 0.026088321],
    [0.630767822,  0.081645948, 0.030617366],
    [0.501098633,  0.087681849, 0.032949235],
    [0.397979736,  0.094769932, 0.036607205],
    [0.316162109,  0.102815928, 0.038563228],
    [0.250976563,  0.109689527, 0.039356409],
    [0.199523926,  0.118074906, 0.037907794],
    [0.15838623,   0.125436013, 0.035678454],
    [0.12588501,   0.131372822, 0.033477382],
    [0.09999,      0.137873613, 0.030135885],
    [0.079439972,  0.141376224, 0.027461742],
    [0.063101413,  0.144377342, 0.024434176],
    [0.050123234,  0.147335978, 0.022420025],
    [0.039814698,  0.148012276, 0.020404185],
    [0.031625623,  0.150982504, 0.020178247],
    [0.025121125,  0.150792897, 0.019329331],
    [0.019954618,  0.153867789, 0.020487372],
    [0.015850358,  0.15635527,  0.01896543],
    [0.012590513,  0.158453721, 0.022077421],
    [0.01,         0.159896748, 0.024543966]
])
_meas_holder3 = np.array([
    [10.0,         0.064310753, 0.003780446],
    [7.943145752,  0.064542767, 0.004498564],
    [6.309448242,  0.065131378, 0.005358283],
    [5.01159668,   0.065125686, 0.006124746],
    [3.980743408,  0.065565601, 0.007402311],
    [3.162109375,  0.066377838, 0.008774717],
    [2.511749268,  0.067277033, 0.011030777],
    [1.995239258,  0.068308829, 0.013284014],
    [1.584655762,  0.069334461, 0.015910647],
    [1.25881958,   0.07112493,  0.018925738],
    [0.999755859,  0.073844257, 0.022281353],
    [0.794250488,  0.077403913, 0.026088321],
    [0.630767822,  0.081645948, 0.030617366],
    [0.501098633,  0.087681849, 0.032949235],
    [0.397979736,  0.094769932, 0.036607205],
    [0.316162109,  0.102815928, 0.038563228],
    [0.250976563,  0.109689527, 0.039356409],
    [0.199523926,  0.118074906, 0.037907794],
    [0.15838623,   0.125436013, 0.035678454],
    [0.12588501,   0.131372822, 0.033477382],
    [0.09999,      0.137873613, 0.030135885],
    [0.079439972,  0.141376224, 0.027461742],
    [0.063101413,  0.144377342, 0.024434176],
    [0.050123234,  0.147335978, 0.022420025],
    [0.039814698,  0.148012276, 0.020404185],
    [0.031625623,  0.150982504, 0.020178247],
    [0.025121125,  0.150792897, 0.019329331],
    [0.019954618,  0.153867789, 0.020487372],
    [0.015850358,  0.15635527,  0.01896543],
    [0.012590513,  0.158453721, 0.022077421],
    [0.01,         0.159896748, 0.024543966]
])

# Initialize the three calibrators ONCE during startup
calibrator_c1 = HolderCalibrator(_ref, _meas_holder1, name="Cell 1")
calibrator_c2 = HolderCalibrator(_ref, _meas_holder2, name="Cell 2")
calibrator_c3 = HolderCalibrator(_ref, _meas_holder3, name="Cell 3")

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

                                # Z1real, Z1imag = calibrator_c1.correct(sfreq, Z1.real, -Z1.imag)
                                # print(f"Cell-1 Impedance: {Z1real} + ({Z1imag}j)")

                                # Z2real, Z2imag = calibrator_c2.correct(sfreq, Z2.real, -Z2.imag)
                                # print(f"Cell-2 Impedance: {Z2real} + ({Z2imag}j)")

                                # Z3real, Z3imag = calibrator_c3.correct(sfreq, Z3.real, -Z3.imag)
                                # print(f"Cell-3 Impedance: {Z3real} + ({Z3imag}j)")

                                Z1real, Z1imag = Z1.real, -Z1.imag
                                Z2real, Z2imag = Z2.real, -Z2.imag
                                Z3real, Z3imag = Z3.real, -Z3.imag

                                # Data Quality Check
                                if i_idx > 0 and ((Z1real < 0.98*sample_c1[i_idx-1, 1] and Z1real < 0) or (Z2real < 0.98*sample_c2[i_idx-1, 1] and Z2real < 0) or (Z3real < 0.98*sample_c3[i_idx-1, 1] and Z3real < 0)):
                                    print("\nFrequency skipped (Impedance Drop)...\n")
                                    break
                                
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